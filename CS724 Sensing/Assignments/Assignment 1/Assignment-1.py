import numpy as np
from sympy import *
import matplotlib.pyplot as plt

# speed of light in m/s
speed_of_light = 3*(10^8)

# Fixing user at (100, 100, 100)
user_position = np.array([100, 100, 100])

# Fixing satellites at random location
satellite_positions = np.array([
    [5000, 1000, 2000],
    [1000, 6000, 7000],
    [3000, 7000, 9000],
    [8000, 2000, 1000],
    [6000, 9000, 3000]
])


print("\nPart (a)")

# Calculating euclidean distance between user and each satellite
distances = np.sqrt(np.sum((satellite_positions - user_position)**2, axis=1))

# Time taken by each satellite to send signal to user
times = distances/speed_of_light

#enumerate adds a counter to each iterable and returns enumerating object
for i, t in enumerate(times):
    print(f"Signal from satellite {i+1} reaches user in {t} seconds")


print("\nPart (b)")

# Creating matrix 'A' to hold coeeficients of x, y, z
A = np.zeros((len(satellite_positions)-1, 3))
# Creating vector 'b' to hold contant terms of RHS
b = np.zeros(len(satellite_positions)-1)

for i in range(1, len(satellite_positions)):
    A[i-1] = 2*(satellite_positions[i] - satellite_positions[0])
    # (r1^2 - r2^2) = c^2*t1^2 - c^2*t2^2
    b[i-1] = ( (speed_of_light**2)*(times[0]**2 - times[i]**2) ) - np.sum(satellite_positions[i]**2 - satellite_positions[0]**2)
    
estimated_user_position = np.linalg.lstsq(A, b, rcond=None)[0]

# User Position without errors
print("\nEstimated User Position (without errors):", estimated_user_position)


print("\nPart (c)")

def addRandomError(times, error_scale):
    return (times + np.random.normal(0, error_scale, len(times)))

# Error = 1ns
error_scale = 1e-9
times_with_error = addRandomError(times, error_scale)

# Estimating position with error
b_with_error = np.zeros(len(satellite_positions) - 1)

for i in range(1, len(satellite_positions)):
    b_with_error[i-1] = ( (speed_of_light**2) * (times_with_error[0]**2 - times_with_error[i]**2) ) - np.sum(satellite_positions[i]**2 - satellite_positions[0]**2)

estimated_user_position_with_error = np.linalg.lstsq(A, b_with_error, rcond=None)[0]

# Calculating localization error
localization_error = np.linalg.norm(user_position-estimated_user_position_with_error)

print("\nEstimated User Position (with small random errors):", estimated_user_position_with_error)
print("Localization Error (meters):", localization_error)


print("\nPart (d)")

# Error scale from 1ns to 100ns
error_scales = np.linspace(1e-9, 1e-7, 50)
localization_errors = []

for scale in error_scales:
    times_with_error = addRandomError(times, scale)
    b_with_error = np.zeros(len(satellite_positions) - 1)

    for i in range(1, len(satellite_positions)):
        b_with_error[i-1] = ( (speed_of_light**2) * (times_with_error[0]**2 - times_with_error[i]**2) - np.sum(satellite_positions[i]**2 - satellite_positions[0]**2) )
    
    estimated_user_position_with_error = np.linalg.lstsq(A, b_with_error, rcond=None)[0]
    localization_error = np.linalg.norm(user_position - estimated_user_position_with_error)
    localization_errors.append(localization_error)

# Plotting the results
plt.plot(error_scales, localization_errors)
plt.xlabel('Timing Error (seconds)')
plt.ylabel('Localozation Error (meters)')
plt.title('Effect of Timing Errors on Localization Accuracy')
plt.grid(True)
plt.show()
