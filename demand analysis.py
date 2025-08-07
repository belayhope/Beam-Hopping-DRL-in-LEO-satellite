import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TrafficDemandSimulator:
    def __init__(self, rows=6, cols=10, a_max=100, seed=None):
        self.rows = rows
        self.cols = cols
        self.num_cells = rows * cols
        self.a_max = a_max
        self.rho = self._generate_normalized_traffic(seed)

    def _generate_normalized_traffic(self, seed):
        if seed is not None:
            np.random.seed(seed)
        rho = np.random.rand(self.num_cells) * 0.2
        high_demand_indices = np.random.choice(self.num_cells, 10, replace=False)
        rho[high_demand_indices] += 0.8
        rho /= rho.max()
        return rho

    def get_arrival_data(self, time_slots=10):
        traffic_matrix = np.zeros((time_slots, self.num_cells))
        for t in range(time_slots):
            D_max = self.a_max * self.rho
            arrivals = np.random.uniform(0, D_max)
            traffic_matrix[t] = arrivals
        return traffic_matrix

    def plot_initial_demand_3d(self):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')

        lat_vals = np.linspace(21.5, 25.3, self.rows)
        lon_vals = np.linspace(119.3, 122.0, self.cols)

        lon, lat = np.meshgrid(lon_vals, lat_vals)
        lon = lon.flatten()
        lat = lat.flatten()
        z = np.zeros_like(lon)
        dx = dy = 0.05 * np.ones_like(lon)
        dz = self.rho * self.a_max  # Scale to actual traffic demand

        cmap = plt.get_cmap('viridis')
        colors = cmap(dz / dz.max())  # Normalize for coloring only

        ax.bar3d(lon, lat, z, dx, dy, dz, color=colors, shade=True)

        ax.set_xlabel('Longitude (°E)', fontsize=14)
        ax.set_ylabel('Latitude (°N)', fontsize=14)
        ax.set_zlabel('Traffic Demand (Mbps)', fontsize=14)

        # View angle adjustment to move Z-axis to the left
        ax.view_init(elev=25, azim=225)

        ax.xaxis.pane.set_alpha(0.2)
        ax.yaxis.pane.set_alpha(0.2)
        ax.zaxis.pane.set_alpha(0.3)
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)

        plt.tight_layout()
        plt.show()

    def plot_initial_demand_2d(self):
        fig, ax = plt.subplots(figsize=(10, 5))

        cell_ids = np.arange(self.num_cells)
        demand = self.rho * self.a_max  # Actual traffic demand
        cmap = plt.get_cmap('viridis')
        colors = cmap(demand / demand.max())

        ax.bar(cell_ids, demand, color=colors, edgecolor='black')

        ax.set_xlabel('Cell ID', fontsize=15)
        ax.set_ylabel('Traffic Demand (Mbps)', fontsize=15)
        #ax.set_title('Traffic Demand vs Cell ID', fontsize=14)
        ax.grid(True)

        plt.tight_layout()
        plt.show()

# Example usage
simulator = TrafficDemandSimulator(a_max=100, seed=42)
simulator.plot_initial_demand_3d()  # 3D bar chart with updated labels
simulator.plot_initial_demand_2d()  # 2D plot: Traffic Demand vs Cell ID

# Print sample traffic arrival
traffic = simulator.get_arrival_data(time_slots=5)
print("Sample traffic arrival (in Mbps) over 5 time slots:")
print(traffic.round(2))
import numpy as np
import matplotlib.pyplot as plt

# Example parameters
num_time_slots = 100
num_satellites = 4

# Simulate some random demand data (you should replace this with your real demand)
np.random.seed(0)
demand = np.random.poisson(lam=50, size=(num_time_slots, num_satellites))

# Initialize array to store running averages
average_demand = np.zeros_like(demand, dtype=float)

# Compute cumulative average for each satellite over time slots
for s in range(num_satellites):
    cumulative_sum = 0
    for t in range(num_time_slots):
        cumulative_sum += demand[t, s]
        average_demand[t, s] = cumulative_sum / (t + 1)  # Average up to time t

# Plot average demand over time for each satellite
plt.figure(figsize=(8,5))
time_slots = np.arange(1, num_time_slots + 1)

for s in range(num_satellites):
    plt.plot(time_slots, average_demand[:, s], label=f'Sat_ {s+1}')

plt.xlabel('Time Slot', fontsize='15')
plt.ylabel('Average Traffic Demand  (Mbits)', fontsize='15')
#plt.title('Average Demand per Satellite over Time Slots')
plt.legend()
plt.grid(True)
plt.show()
