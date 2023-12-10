import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def animate_particle_filter(landmark_map, odometry, noisy_observations, particle_history, estimated_poses):
    fig, ax = plt.subplots()
    landmark_plot, = ax.plot(landmark_map[:, 0], landmark_map[:, 1], 'bo', label='Landmarks')
    odometry_plot, = ax.plot([], [], 'r-', label='Odometry')
    estimated_pose_plot, = ax.plot([], [], 'k-', label='Estimated Pose')
    particles_plot = ax.scatter([], [], s=1, color='gray', label='Particles')
    noisy_observation_plot = ax.scatter([], [], s=30, color='r', marker='x', label='Noisy Observations')

    def init():
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        return landmark_plot, odometry_plot, estimated_pose_plot, particles_plot, noisy_observation_plot

    def update(frame):
        odometry_plot.set_data(odometry[:frame, 0], odometry[:frame, 1])
        estimated_pose_plot.set_data(estimated_poses[:frame, 0], estimated_poses[:frame, 1])
        particles = particle_history[frame]
        particles_plot.set_offsets(particles[:, :2])

        noisy_obs = noisy_observations[frame]
        if len(noisy_obs) > 0 and noisy_obs.ndim == 2:
            noisy_observation_plot.set_offsets(noisy_obs[:, :2])
        else:
            noisy_observation_plot.set_offsets(np.empty((0, 2)))  # Use an empty 2D array for no observations

        return landmark_plot, odometry_plot, estimated_pose_plot, particles_plot, noisy_observation_plot


    ani = animation.FuncAnimation(fig, update, frames=len(estimated_poses), init_func=init, blit=True, interval=50)
    plt.legend()
    plt.show()
    # Uncomment the following line to save the animation as a video file
    ani.save('video1/particles_0_0_H_200.mp4', writer='ffmpeg')

def particle_filter(map_path, sensing_path, num_particles, estimates_path):
    odometry = []
    noisy_observations = []
    particle_history = []
    landmark_map = np.load(map_path)
    readings = np.load(sensing_path)
    initial_pose = readings[0, :3]

    particles = initialize_particles(num_particles, initial_pose)
    estimates = np.zeros((len(readings) // 2, 3))

    print("Readings array shape:", readings.shape)
    print("Sample readings:", readings[:5])  # Print first few rows for inspection

    for i in range(1, len(readings), 2):
        particles = update_particles(particles, readings[i])
        weights = weight_particles(particles, readings[i + 1], landmark_map)

        odometry.append(readings[i, :2])
    
        # Reshape noisy observations for each frame into a 2D array
        noisy_obs_frame = readings[i + 1].reshape(-1, 2)  # Assuming each landmark observation has 2 values
        noisy_observations.append(noisy_obs_frame)
    
        particles = resample_particles(particles, weights)

        # Estimate robot pose as the mean of the particle cloud
        estimates[i // 2] = np.average(particles, axis=0, weights=weights)

        odometry.append(readings[i, :2])
        noisy_observations.append(readings[i + 1, :])
        particle_history.append(particles.copy())

    np.save(estimates_path, estimates)

    # Return odometry and noisy_observations as lists
    return np.array(odometry), noisy_observations, particle_history, estimates



if __name__ == "__main__":
    args = parse_arguments()
    odometry, noisy_obs, particle_history, estimated_poses = particle_filter(args.map, args.sensing, args.num_particles, args.estimates)
    landmark_map = np.load(args.map)
    animate_particle_filter(landmark_map, odometry, noisy_obs, particle_history, estimated_poses)
