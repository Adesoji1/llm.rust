import matplotlib.pyplot as plt
import numpy as np

# Data
B_values = [4, 8, 16, 32, 64, 128]
input_sizes = [12288, 98304, 786432, 6291456, 50331648, 402653184]

# Matmul Forward Times (ms)
forward_times = [
    [1.20, 0.48437, 0.16891],
    [0.27997, 0.28479, 0.33879],
    [0.49162, 0.49637, 0.58741],
    [5.32, 4.92, 5.58],
    [41.41, 40.79, 42.29],
    [588.21, 589.25, 591.60],
]

# Matmul Backward Times (ms)
backward_times = [
    [0.02445, 0.02583, 0.02287],
    [0.18479, 0.181, 0.229],
    [1.28, 1.18, 1.45],
    [14.32, 15.05, 14.00],
    [182.69, 183.82, 183.28],
    [2510, 2500, 2610],
]

# Calculate means and standard deviations
def calculate_stats(times_list):
    means = []
    stds = []
    for times in times_list:
        mean = np.mean(times)
        std = np.std(times)
        means.append(mean)
        stds.append(std)
    return means, stds

forward_means, forward_stds = calculate_stats(forward_times)
backward_means, backward_stds = calculate_stats(backward_times)

print(forward_means, forward_stds)
print(backward_means, backward_stds)
stop
# Plotting Matmul Forward Results
plt.figure(figsize=(10, 6))
plt.errorbar(B_values, forward_means, yerr=forward_stds, fmt='-o', capsize=5)
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Batch Size (B)')
plt.ylabel('Time (ms)')
plt.title('Matmul Forward Performance')
plt.grid(True, which="both", ls="--")
plt.xticks(B_values, B_values)
#plt.show()
plt.savefig("matmul_forward_performance.png", dpi=300)

# Plotting Matmul Backward Results
plt.figure(figsize=(10, 6))
plt.errorbar(B_values, backward_means, yerr=backward_stds, fmt='-o', color='red', capsize=5)
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Batch Size (B)')
plt.ylabel('Time (ms)')
plt.title('Matmul Backward Performance')
plt.grid(True, which="both", ls="--")
plt.xticks(B_values, B_values)
#plt.show()
plt.savefig("matmul_backward_performance.png", dpi=300)
