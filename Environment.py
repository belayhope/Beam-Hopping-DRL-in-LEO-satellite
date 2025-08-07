# -*- coding: utf-8 -*-
"""
@author: Belayneh Abebe
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
from gymnasium import spaces
import gym
# Constants
NUM_SATELLITES = 4
BEAMS_PER_SAT = 6
NUM_CELLS = 60
TOTAL_BANDWIDTH = 500e6  # 500 MHz
BANDWIDTH_PER_CHUNK = 62.5e6
CARRIER_FREQ = 30e9  # 30 GHz
ALTITUDE = 550e3  # 550 km
BOLTZMANN = 1.38e-23
NOISE_TEMP = 300
TIME_SLOT = 0.002  # 2 ms
EIRP_DBM = 43
BEAMWIDTH = 3.2  # degrees
GAMMA = 0.99
LAMBDA = 0.95

class SatelliteBHEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Corrected obs_dim calculation to match _get_obs() output size
        obs_dim = NUM_CELLS + NUM_CELLS + (NUM_SATELLITES * BEAMS_PER_SAT) + (NUM_SATELLITES * BEAMS_PER_SAT)
        self.observation_space = spaces.Box(low=0, high=1e3, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Dict({
            "beam_angle": spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
            "bandwidth": spaces.Discrete(int(TOTAL_BANDWIDTH // BANDWIDTH_PER_CHUNK)),
            "power": spaces.Box(low=0, high=EIRP_DBM, shape=(1,), dtype=np.float32),
            "precoding_index": spaces.Discrete(8)
        })

    def reset(self):
        self.demand = np.random.uniform(0.5, 5, size=(NUM_CELLS,))
        self.buffer = np.copy(self.demand)
        self.cell_positions = np.random.rand(NUM_CELLS, 2) * 1000
        self.beam_positions = np.random.rand(NUM_SATELLITES * BEAMS_PER_SAT, 2) * 1000
        self.cell_map = {i: i % NUM_CELLS for i in range(NUM_SATELLITES * BEAMS_PER_SAT)}
        self.powers = np.zeros(NUM_SATELLITES * BEAMS_PER_SAT)
        self.bandwidths = np.zeros(NUM_SATELLITES * BEAMS_PER_SAT)
        self.agent_index = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.demand,
            self.buffer,
            self.powers,
            self.bandwidths
        ]).astype(np.float32)

    def _channel_gain(self, cell_position, beam_position):
        d = np.linalg.norm(np.array(cell_position) - np.array(beam_position))
        fspl_db = 20 * np.log10(d) + 20 * np.log10(CARRIER_FREQ) + 92.45
        gain = EIRP_DBM + 30 - fspl_db
        return gain

    def _calculate_snr(self, power_dbm, gain_db):
        power_watt = 10 ** ((power_dbm - 30) / 10)
        noise_watt = BOLTZMANN * NOISE_TEMP * BANDWIDTH_PER_CHUNK
        snr = power_watt * 10 ** (gain_db / 10) / noise_watt
        return 10 * np.log10(snr)

    def step(self, action):
        idx = self.agent_index % (NUM_SATELLITES * BEAMS_PER_SAT)
        cell_idx = self.cell_map[idx]

        beam_angle = action["beam_angle"][0]
        bw_idx = action["bandwidth"]
        power = action["power"][0]
        precoding_index = action["precoding_index"]

        self.bandwidths[idx] = bw_idx * BANDWIDTH_PER_CHUNK
        self.powers[idx] = power

        gain_db = self._channel_gain(self.cell_positions[cell_idx], self.beam_positions[idx])
        snr_db = self._calculate_snr(power, gain_db)

        rate = BANDWIDTH_PER_CHUNK * np.log2(1 + 10 ** (snr_db / 10)) / 1e9
        satisfied = min(rate, self.buffer[cell_idx])
        reward = satisfied
        self.buffer[cell_idx] -= satisfied

        # Set done to True if the cell's buffer is depleted
        done = self.buffer[cell_idx] <= 0

        self.agent_index += 1

        return self._get_obs(), reward, done, {}

    def render(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.cell_positions[:, 0], self.cell_positions[:, 1], c=self.buffer, cmap='Reds', label='Cells')
        plt.scatter(self.beam_positions[:, 0], self.beam_positions[:, 1], marker='^', label='Beams')
        plt.colorbar(label='Remaining Buffer (Gbps)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Beam and Cell Positions with Buffer Levels')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()