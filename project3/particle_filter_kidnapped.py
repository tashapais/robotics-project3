import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
import os
import argparse
from dead_reckoning import drive_model, rotate, transform_rigid_body, to_global
from matplotlib.animation import FuncAnimation

# For animation
dt = 0.1
original = np.array([[-.05, -.1], [.05, -.1], [.05, .1], [-.05, .1]])

# Define the ranges for particle initialization
xlim = [0, 2]  # Assuming the x-coordinates range from 0 to 2
ylim = [0, 2]  # Assuming the y-coordinates range from 0 to 2
theta_range = [0, 2*np.pi]  # Assuming theta (orientation) ranges from 0 to 2*pi


# Initialization with uniform distribution
def initialize_particles_uniform(num_particles, xlim, ylim, theta_range):
    x = np.random.uniform(xlim[0], xlim[1], num_particles)
    y = np.random.uniform(ylim[0], ylim[1], num_particles)
    theta = np.random.uniform(theta_range[0], theta_range[1], num_particles)
    return np.vstack((x, y, theta)).T

# Prediction
def predict(particles, control, dt, std):
    N = len(particles)
    # Add noise to the control input
    noisy_control = control + np.random.randn(N, 2) * std
    # Update particles based on motion model
    particles[:, 0] += noisy_control[:, 0] * np.cos(particles[:, 2]) * dt
    particles[:, 1] += noisy_control[:, 0] * np.sin(particles[:, 2]) * dt
    particles[:, 2] += noisy_control[:, 1] * dt
    return particles

# Update
def update(particles, weights, measurement, std, landmarks):
    num_landmarks = landmarks.shape[0]
    for i in range(len(particles)):
        distances = np.linalg.norm(landmarks - particles[i, :2], axis=1)
        angles = np.arctan2(landmarks[:, 1] - particles[i, 1], landmarks[:, 0] - particles[i, 0]) - particles[i, 2]

        weight = 1.0
        for j in range(num_landmarks):
            if measurement.ndim == 1:
                # Handle one-dimensional measurement
                distance_measurement = measurement[0]
                angle_measurement = measurement[1]
            else:
                # Handle two-dimensional measurement
                distance_measurement = measurement[j, 0]
                angle_measurement = measurement[j, 1]

            weight *= norm(distances[j], std[0]).pdf(distance_measurement)
            weight *= norm(angles[j], std[1]).pdf(angle_measurement)

        weights[i] *= weight

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize
    return weights


# Resample
def resample(particles, weights):
    indices = np.random.choice(len(particles), len(particles), p=weights)
    return particles[indices]

# Particle Filter
def particle_filter(initial_pose, controls, measurements, num_particles, landmarks, std, xlim, ylim, theta_range):
    particles = initialize_particles_uniform(num_particles, xlim, ylim, theta_range)
    weights = np.ones(num_particles) / num_particles
    estimates = np.zeros((len(controls) + 1, 3))  # +1 to include initial pose
    estimates[0] = initial_pose  # Set the first row to the initial pose

    for i, control in enumerate(controls):
        particles = predict(particles, control, dt=0.1, std=std[0])
        weights = update(particles, weights, measurements, std[1], landmarks)
        particles = resample(particles, weights)
        estimates[i + 1] = np.mean(particles, axis=0)  # +1 to account for initial pose

    return estimates

def parse_arguments():
    parser = argparse.ArgumentParser(description='Particle Filter for Robot Localization')
    parser.add_argument('--map', required=True, help='Path to landmark map file')
    parser.add_argument('--sensing', required=True, help='Path to sensor readings file')
    parser.add_argument('--num_particles', type=int, required=True, help='Number of particles to use in the filter')
    parser.add_argument('--estimates', required=True, help='Path to store the estimates')
    return parser.parse_args()

# Update function for animation
def update_frame(frame_number, controls, measurements, std, landmarks, scat):
    global particles, weights

    # Check if we have reached the end of controls
    if frame_number < len(controls):
        control = controls[frame_number]
        measurement = measurements[frame_number]

        # Predict, Update, Resample
        particles = predict(particles, control, dt=0.1, std=std[0])
        weights = update(particles, weights, measurement, std[1], landmarks)
        particles = resample(particles, weights)

    # Update scatter plot
    scat.set_offsets(particles[:, :2])

if __name__ == "__main__":
    args = parse_arguments()
    num_particles = args.num_particles
    landmarks = np.load(args.map, allow_pickle=True)
    readings = np.load(args.sensing, allow_pickle=True)

    initial_pose = readings[0, :3]
    controls = readings[1::2, :2]
    measurements = readings[2::2, :]
    std_control = np.array([0.05, 0.05])
    std_measurement = np.array([0.02, 0.02])
    std = [std_control, std_measurement]

    # Run particle filter for estimates
    estimates = particle_filter(initial_pose, controls, measurements, num_particles, landmarks, std, xlim, ylim, theta_range)
    np.save(args.estimates, estimates)  # Save the estimates as a .npy file

    # Animation Setup
    fig, ax = plt.subplots()
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    particles = initialize_particles_uniform(num_particles, xlim, ylim, theta_range)  # Corrected function call
    weights = np.ones(num_particles) / num_particles
    scat = ax.scatter(particles[:, 0], particles[:, 1], color='grey', s=10)

    # Create and run animation
    ani = FuncAnimation(fig, update_frame, fargs=(controls, measurements, std, landmarks, scat),
                        frames=len(controls), interval=50, repeat=False)

    # Save animation
    ani.save('video2/particles_0_0_L_200.mp4', writer='ffmpeg', fps=30)

    plt.show()
