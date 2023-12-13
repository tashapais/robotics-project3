# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # Define the paths and parameters
# landmark_maps = ["maps/landmark_1.npy", "maps/landmark_2.npy", "maps/landmark_3.npy", "maps/landmark_4.npy", "maps/landmark_5.npy"]
# first_numbers = range(5)
# second_numbers = range(2)
# third_letters = ["H", "L"]
# noise_levels = [200, 1000]

# # Iterate through all combinations
# for first_num in first_numbers:
#     for second_num in second_numbers:
#         for third_letter in third_letters:
#             for noise_level in noise_levels:
#                 # Generate control sequence filename
#                 reading_filename = f"readings/readings_{first_num}_{second_num}_{third_letter}.npy"

#                 # Generate a unique output filename based on the combination
#                 output_filename = f"estim1/{first_num}_{second_num}_{third_letter}{noise_level}.npy"

#                 # Run the particle filter for the current combination
#                 os.system(f"python particle_filter.py --map {landmark_maps[first_num]} --sensing {reading_filename} --num_particles {noise_level} --estimates {output_filename}")

import numpy as np
import matplotlib.pyplot as plt

# Load the estimated poses
estimated_poses = np.load('estim1/estim1_0_1_L_500.npy')

# Load the ground truth poses (you should replace this with the correct path to your ground truth poses)
# ground_truth_poses = np.load('path_to_ground_truth_poses.npy')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(estimated_poses[:, 0], estimated_poses[:, 1], label='Estimated Trajectory', color='r')
# plt.plot(ground_truth_poses[:, 0], ground_truth_poses[:, 1], label='Ground Truth Trajectory', color='g')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Particle Filter Estimated Trajectory vs Ground Truth')
plt.legend()
plt.show()

# Calculate the position error
position_error = np.sqrt(np.sum((estimated_poses[:, :2] - ground_truth_poses[:, :2])**2, axis=1))

# Calculate the orientation error (adjust this calculation based on how your angles are represented)
orientation_error = np.abs(estimated_poses[:, 2] - ground_truth_poses[:, 2])

# Plot the errors
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(position_error, label='Position Error')
plt.ylabel('Position Error')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(orientation_error, label='Orientation Error')
plt.xlabel('Time Step')
plt.ylabel('Orientation Error')
plt.legend()

plt.show()

