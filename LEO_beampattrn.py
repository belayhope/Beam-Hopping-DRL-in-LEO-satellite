# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 13:50:01 2025

@author: Belayneh Abebe
"""
# antenna beam pattren in LEO satellite 
import numpy as np
import matplotlib.pyplot as plt

# Constants based on your description
G_m = 38.5  # Example max gain in dBi, adjust as needed
theta_b = 0.65  # Half of 3 dB beamwidth in degrees, example value
Ls = -6.75  # Near-in side-lobe level in dBi
Y = 1.5 * theta_b
Z = Y + 0.04**(G_m + Ls)  # Using your formula, corrected below

# The original formula for Z was: Z = Y + 0.04^(G_m + Ls - Lf), Lf = 0 here
# 0.04^(G_m + Ls) is very small, instead likely means 0.04 * (G_m + Ls - Lf)
# So let's clarify and implement as:
Lf = 0
Z = Y + 0.04 * (G_m + Ls - Lf)

# Define theta range from 0 to 2 * Z to see full pattern
theta = np.linspace(0, 2 * Z, 1000)

def antenna_gain(theta_deg):
    G = np.zeros_like(theta_deg)
    for i, t in enumerate(theta_deg):
        if t <= theta_b:
            G[i] = G_m
        elif theta_b < t <= Y:
            G[i] = G_m - 3 * ((t - theta_b) / theta_b)**2
        elif Y < t <= Z:
            G[i] = G_m + Ls - 25 * np.log10(t / theta_b)
        else:
            G[i] = 0
    return G

G_theta = antenna_gain(theta)

plt.figure(figsize=(8, 5))
plt.plot(theta, G_theta, label='Antenna Gain Pattern')
plt.axvline(x=theta_b, color='r', linestyle='--', label=r'$\theta_b$')
plt.axvline(x=Y, color='g', linestyle='--', label='Y')
plt.axvline(x=Z, color='b', linestyle='--', label='Z')
plt.xlabel(r'$\theta$ (degrees)',fontsize=14)
plt.ylabel('Gain (dBi)',fontsize=14)
#plt.title('Multi-beam Antenna Radiation Pattern (ITU-R S.1528)')
plt.legend()
plt.grid(True)
plt.show()
