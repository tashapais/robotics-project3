import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from dead_reckoning import drive_model, rotate, transform_rigid_body, to_global
from matplotlib.animation import FuncAnimation

dt = 0.1
original = np.array([[-.05, -.1], [.05, -.1], [.05, .1], [-.05, .1]])

def parse_arguments():
    parser = argparse.ArgumentParser(description='Particle Filter for Robot Localization')
    parser.add_argument('--map', required=True, help='Path to landmark map file')
    parser.add_argument('--sensing', required=True, help='Path to sensor readings file')
    parser.add_argument('--num_particles', type=int, required=True, help='Number of particles to use in the filter')
    parser.add_argument('--estimates', required=True, help='Path to store the estimates')
    return parser.parse_args()

def initialize_particles(num_particles, initial_pose):
    particles = np.tile(initial_pose, (num_particles, 1))
    return particles

def update_particles(particles, control, dt=0.1):
    # Add noise to control
    # Only generate noise for the control inputs (linear and angular velocities)
    control_noise = np.random.normal(0, [0.05, 0.1], size=(len(particles), 2))
    noisy_control = control[:2] + control_noise  # Add noise to control inputs

    # Update particle positions
    particles[:, 0] += noisy_control[:, 0] * np.cos(particles[:, 2]) * dt
    particles[:, 1] += noisy_control[:, 0] * np.sin(particles[:, 2]) * dt
    particles[:, 2] += noisy_control[:, 1] * dt
    return particles

def weight_particles(particles, measurement, landmarks):
    weights = np.ones(len(particles))
    
    for i, landmark in enumerate(landmarks):
        # Calculate expected distance and angle to each landmark from each particle
        expected_distance = np.sqrt((particles[:, 0] - landmark[0])**2 + (particles[:, 1] - landmark[1])**2)
        expected_angle = np.arctan2(landmark[1] - particles[:, 1], landmark[0] - particles[:, 0]) - particles[:, 2]

        # Extract the actual distance and angle measurements for the ith landmark
        actual_distance = measurement[i*2]  # Even indices are distance measurements
        actual_angle = measurement[i*2 + 1]  # Odd indices are angle measurements

        # Calculate weights based on Gaussian noise model
        weights *= np.exp(-(expected_distance - actual_distance)**2 / 0.04)
        weights *= np.exp(-(expected_angle - actual_angle)**2 / 0.04)
    return weights


def resample_particles(particles, weights):
    # Normalize weights to prevent division by zero or NaN values
    weights += 1e-300  # add a small value to avoid division by zero
    normalized_weights = weights / np.sum(weights)

    # Check for NaN values in the weights
    if np.isnan(normalized_weights).any():
        print("NaN values found in weights, resampling may not work correctly.")
        normalized_weights = np.nan_to_num(normalized_weights)  # Replace NaNs with zeros

    indices = np.random.choice(len(particles), size=len(particles), p=normalized_weights)
    return particles[indices]

# def weight_particles(particles, measurement, landmarks):
#     weights = np.zeros(len(particles))
    
#     for i, landmark in enumerate(landmarks):
#         # Calculate expected distance and angle to each landmark from each particle
#         expected_distance = np.sqrt((particles[:, 0] - landmark[0])**2 + (particles[:, 1] - landmark[1])**2)
#         expected_angle = np.arctan2(landmark[1] - particles[:, 1], landmark[0] - particles[:, 0]) - particles[:, 2]

#         # Extract the actual distance and angle measurements for the ith landmark
#         actual_distance = measurement[i*2]  # Even indices are distance measurements
#         actual_angle = measurement[i*2 + 1]  # Odd indices are angle measurements

#         # Calculate weights using log-likelihoods to avoid underflow
#         log_weight_distance = -(expected_distance - actual_distance)**2 / (2 * 0.04)
#         log_weight_angle = -(expected_angle - actual_angle)**2 / (2 * 0.04)
#         weights += log_weight_distance + log_weight_angle

#     # Convert log weights to actual weights
#     max_log_weight = np.max(weights)
#     weights = np.exp(weights - max_log_weight)

#     return weights

# def resample_particles(particles, weights):
#     # Normalize weights to prevent division by zero or NaN values
#     weights += 1e-300  # Add a small value to avoid division by zero
#     normalized_weights = weights / np.sum(weights)

#     indices = np.random.choice(len(particles), size=len(particles), p=normalized_weights)
#     resampled_particles = particles[indices]
#     return resampled_particles

def particle_filter(map_path, sensing_path, num_particles, estimates_path):
    landmarks = np.load(map_path)
    readings = np.load(sensing_path)
    initial_pose = readings[0, :3]

    particles = initialize_particles(num_particles, initial_pose)
    
    # Initialize estimates array with an additional row for the initial estimate
    estimates = np.zeros((len(readings) // 2 + 1, 3))
    
    # Set the first row of estimates to the initial pose
    estimates[0] = np.mean(particles, axis=0)

    for i in range(1, len(readings), 2):
        control = readings[i][:2]  # Ensure control is taken correctly
        measurement = readings[i + 1]

        particles = update_particles(particles, control, dt)

        print("Weights before resampling:", weights)
        weights = weight_particles(particles, measurement, landmarks)
        print("Weights after resampling:", weights)

        particles = resample_particles(particles, weights)

        # Update estimates after each resampling
        estimates[(i // 2) + 1] = np.mean(particles, axis=0)

    np.save(estimates_path, estimates)
    return estimates

def animate(i, q, particles, particles_plot, ground_truths, readings, landmarks, gt_patch, sensed_patch, landmark_guesses, lines, ax):
    global odometry_plot, estimated_pose_plot, noisy_observation_plot

    # Update particles and resample
    particles = update_particles(particles, readings[i * 2 + 1])
    weights = weight_particles(particles, readings[i * 2 + 2], landmarks)
    particles = resample_particles(particles, weights)
    particles_plot.set_offsets(particles[:, :2])  # Update particle positions


    # Update estimated pose plot (black line)
    estimated_pose = np.mean(particles, axis=0)
    estimated_pose_x, estimated_pose_y = estimated_pose_plot.get_data()
    estimated_pose_x = np.append(estimated_pose_x, estimated_pose[0])
    estimated_pose_y = np.append(estimated_pose_y, estimated_pose[1])
    estimated_pose_plot.set_data(estimated_pose_x, estimated_pose_y)

    # Update ground truth and sensed patches
    gt_patch.set_xy(transform_rigid_body(original, ground_truths[i + 1]))
    q += drive_model(q, readings[(2 * i) + 1])
    sensed_patch.set_xy(transform_rigid_body(original, q))

    # Plot odometry and estimated poses
    ax.plot(ground_truths[i][0], ground_truths[i][1], 'o', color='blue', markersize=1.5)
    ax.plot(q[0], q[1], 'o', color='red', markersize=1.5)

    # Update landmark guesses
    sensed_landmarks = to_global(readings[(i * 2) + 2], q)
    landmark_guesses.set_offsets(sensed_landmarks)

    # Update lines connecting estimated robot position to landmarks
    for j, line in enumerate(lines):
        if j < len(sensed_landmarks):
            line.set_data([q[0], sensed_landmarks[j][0]], [q[1], sensed_landmarks[j][1]])
        else:
            line.set_data([], [])

    return particles_plot, odometry_plot, estimated_pose_plot, noisy_observation_plot, gt_patch, sensed_patch, landmark_guesses, *lines


def load_ground_truth(sensing_path):
    # Extract the first and second numbers from the sensing_path
    parts = os.path.basename(sensing_path).split('_')
    first_number = parts[1]
    second_number = parts[2].split('.')[0]
    
    # Construct the path to the corresponding ground truth file
    ground_truth_path = f'gts/gt_{first_number}_{second_number}.npy'

    # Load the ground truth data
    ground_truths = np.load(ground_truth_path)

    return ground_truths

def construct_animation_save_path(estimates_path):
    # Extract components from the estimates path
    dir_name, file_name = os.path.split(estimates_path)
    dir_name= 'video1'
    parts = file_name.split('_')
    map_number = parts[1]
    control_number = parts[2]
    noise_level = parts[3]
    particle_number = parts[4].split('.')[0]

    # Construct the animation save path
    animation_file_name = f'particles_{map_number}_{control_number}_{noise_level}_{particle_number}.mp4'
    animation_save_path = os.path.join(dir_name, animation_file_name)

    return animation_save_path

if __name__ == "__main__":
    args = parse_arguments()
    ground_truths = load_ground_truth(args.sensing)

    # Load data
    landmarks = np.load(args.map, allow_pickle=True)
    readings = np.load(args.sensing, allow_pickle=True)
    q = np.array(readings[0])
    num_particles = args.num_particles
    initial_pose = readings[0, :3]  # Assuming the initial pose is in the first row

    # Initialize particles
    particles = initialize_particles(num_particles, initial_pose)

    # Set up the plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, 2)  # Adjust as per your map's dimensions
    ax.set_ylim(0, 2)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], marker='o', color='blue', label='Landmarks')
    particles_plot = ax.scatter(particles[:, 0], particles[:, 1], s=1, color='grey', label='Particles')

    odometry_plot, = ax.plot([], [], 'r-', label='Odometry')
    estimated_pose_plot, = ax.plot([], [], 'k-', label='Estimated Pose')
    noisy_observation_plot = ax.scatter([], [], s=30, color='r', marker='x', label='Noisy Observations')

    gt_patch = plt.Polygon(transform_rigid_body(original, ground_truths[0]), fill=False, color='blue')
    sensed_patch = plt.Polygon(transform_rigid_body(original, readings[0]), fill=False, color='red')

    ax.add_patch(gt_patch)
    ax.add_patch(sensed_patch)

    landmark_guesses = ax.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', color='red', s=5)
    lines = [ax.plot([], [], 'r-', linewidth=0.5)[0] for _ in range(len(landmarks))]

    # Create and run the animation
    ani = FuncAnimation(fig, animate, frames=len(readings) // 2, fargs=(q, particles, particles_plot, ground_truths, readings, landmarks, gt_patch, sensed_patch, landmark_guesses, lines, ax), repeat=False)

    plt.legend()
    plt.show()

    # Save particle estimates
    np.save(args.estimates, particles)  
    animation_save_path = construct_animation_save_path(args.estimates)

    #print(animation_save_path)
    # Use this path to save the animation
    # ani.save(animation_save_path, writer='ffmpeg')

