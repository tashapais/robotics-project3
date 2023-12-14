import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from numpy.random import seed
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
import os
import argparse
from dead_reckoning import drive_model, rotate, transform_rigid_body, to_global
from matplotlib.animation import FuncAnimation

dt = 0.1
original = np.array([[-.05, -.1], [.05, -.1], [.05, .1], [-.05, .1]])

def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles

def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles

def predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""

    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist

def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)

def neff(weights):
    return 1. / np.sum(np.square(weights))

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill (1.0 / len(weights))

def animate(i, particles_plot, estimates, ground_truths, readings, landmarks, gt_patch, sensed_patch, landmark_guesses, lines, ax):
    global odometry_plot, estimated_pose_plot, noisy_observation_plot

    # Update the estimated pose plot (black line)
    estimated_pose = estimates[i]
    estimated_pose_x, estimated_pose_y = estimated_pose_plot.get_data()
    estimated_pose_x = np.append(estimated_pose_x, estimated_pose[0])
    estimated_pose_y = np.append(estimated_pose_y, estimated_pose[1])
    estimated_pose_plot.set_data(estimated_pose_x, estimated_pose_y)

    # Update ground truth and sensed patches
    gt_patch.set_xy(transform_rigid_body(original, ground_truths[i]))
    q = ground_truths[i]  # Assuming ground_truths[i] has the format [x, y, theta]
    sensed_patch.set_xy(transform_rigid_body(original, q))

    # Update landmark guesses based on current readings
    sensed_landmarks = to_global(readings[i * 2 + 2], q)
    landmark_guesses.set_offsets(sensed_landmarks)

    # Update lines connecting estimated robot position to landmarks
    for j, line in enumerate(lines):
        if j < len(sensed_landmarks):
            line.set_data([estimated_pose[0], sensed_landmarks[j][0]], [estimated_pose[1], sensed_landmarks[j][1]])
        else:
            line.set_data([], [])

    # Plot odometry and estimated poses
    ax.plot(q[0], q[1], 'o', color='blue', markersize=1.5)
    ax.plot(estimated_pose[0], estimated_pose[1], 'o', color='red', markersize=1.5)

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Particle Filter for Robot Localization')
    parser.add_argument('--map', required=True, help='Path to landmark map file')
    parser.add_argument('--sensing', required=True, help='Path to sensor readings file')
    parser.add_argument('--num_particles', type=int, required=True, help='Number of particles to use in the filter')
    parser.add_argument('--estimates', required=True, help='Path to store the estimates')
    return parser.parse_args()

# def run_particle_filter(args):
#     N = args.num_particles
#     landmarks = np.load(args.map, allow_pickle=True)
#     readings = np.load(args.sensing, allow_pickle=True)
#     initial_pose = readings[0]
#     iters = len(readings) // 2

#     # Initialize the particles
#     particles = create_gaussian_particles(mean=initial_pose, std=(0.2, 0.2, np.pi / 4), N=N)
#     weights = np.ones(N) / N

#     # Initialize variables for tracking
#     estimates = []

#     for i in range(iters):
#         # Control input (u) from the readings
#         u = readings[i * 2 + 1]  # Make sure this is the correct format for control input

#         # Predict the particles' movement
#         predict(particles, u=u, std=(0.2, 0.05))

#         # Sensor measurements (z) from the readings
#         z = readings[i * 2 + 2]  # Ensure this is the correct format for sensor measurements

#         # Update the particles with the measurement
#         update(particles, weights, z=z, R=0.1, landmarks=landmarks)

#         # Resample if needed
#         if neff(weights) < N / 2:
#             indexes = systematic_resample(weights)
#             resample_from_index(particles, weights, indexes)

#         # Estimate the current state
#         mu, var = estimate(particles, weights)
#         estimates.append(mu)

#     # Save the estimated positions
#     np.save(args.estimates, estimates)

def run_particle_filter(args):
    N = args.num_particles
    landmarks = np.load(args.map, allow_pickle=True)
    readings = np.load(args.sensing, allow_pickle=True)
    controls = np.load('controls/controls_0_0.npy')  # Load controls
    initial_pose = readings[0]
    iters = len(controls)  # Number of iterations based on the number of control inputs

    # Initialize the particles
    particles = create_gaussian_particles(mean=initial_pose, std=(0.2, 0.2, np.pi / 4), N=N)
    weights = np.ones(N) / N

    # Initialize variables for tracking
    estimates = []

    for i in range(iters):
        # Use the i-th control input to predict the particles' movement
        u = controls[i]
        predict(particles, u=u, std=(0.2, 0.05))

        # Sensor measurements (z) from the readings corresponding to the i-th time step
        z = readings[i * 2 + 2]  # Adjust this if the structure of the readings file is different

        # Update the particles with the measurement
        update(particles, weights, z=z, R=0.1, landmarks=landmarks)

        # Resample if needed
        if neff(weights) < N / 2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)

        # Estimate the current state
        mu, var = estimate(particles, weights)
        estimates.append(mu)

    # Save the estimated positions
    np.save(args.estimates, estimates)

if __name__ == "__main__":
    args = parse_arguments()

    # Run the particle filter to compute estimates
    run_particle_filter(args)

    # Load computed estimates
    estimates = np.load(args.estimates)
    
    # Load ground truth data and other necessary information
    ground_truths = load_ground_truth(args.sensing)
    landmarks = np.load(args.map, allow_pickle=True)
    readings = np.load(args.sensing, allow_pickle=True)

    # Set up the plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, 2)  # Adjust as per your map's dimensions
    ax.set_ylim(0, 2)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], marker='o', color='blue', label='Landmarks')

    # Initialize plot elements for animation
    particles_plot = ax.scatter([], [], s=1, color='grey', label='Particles')
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
    ani = FuncAnimation(fig, animate, frames=len(readings) // 2,
                        fargs=(particles_plot, estimates, ground_truths, readings, landmarks, gt_patch, sensed_patch, landmark_guesses, lines, ax),
                        repeat=False)

    plt.legend()
    plt.show()

    # Save the animation
    animation_save_path = construct_animation_save_path(args.estimates)
    print(animation_save_path)
    ani.save(animation_save_path, writer='ffmpeg')

