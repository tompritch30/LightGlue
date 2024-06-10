import numpy as np
import matplotlib.pyplot as plt

# Load epipolar errors
epipolar_errors = np.loadtxt("epipolar_errors.txt", delimiter=",")

# Plot histogram
plt.hist(epipolar_errors, bins=50, alpha=0.75, color='blue', edgecolor='black')
plt.title('Epipolar Error Distribution')
plt.xlabel('Epipolar Error')
plt.ylabel('Frequency')
plt.show()

# Determine thresholds based on distribution
low_threshold = np.percentile(epipolar_errors, 50)   # 50th percentile
medium_threshold = np.percentile(epipolar_errors, 90)  # 90th percentile

print(f"Low Threshold: {low_threshold}")
print(f"Medium Threshold: {medium_threshold}")

# Define a function to map error to color
def get_error_color(error):
    if error < low_threshold:
        return 'green'
    elif low_threshold <= error < medium_threshold:
        return 'yellow'
    else:
        return 'red'