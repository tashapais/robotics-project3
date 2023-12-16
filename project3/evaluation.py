import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import os
import sys

original = np.array([[-.05, -.1], [.05, -.1], [.05, .1], [-.05, .1]])

def drive_model(q, u):
    dq = np.zeros_like(q)
    dq[0] = u[0] * np.cos(q[2]) * dt
    dq[1] = u[0] * np.sin(q[2]) * dt
    dq[2] = u[1] * dt
    return dq

def rotate(origin, point, angle):
    newpointx = origin[0] + np.cos(angle) * (point[0] - origin[0]) - np.sin(angle) * (point[1] - origin[1])
    newpointy = origin[1] + np.sin(angle) * (point[0] - origin[0]) + np.cos(angle) * (point[1] - origin[1])
    return [newpointx, newpointy]

def transform_rigid_body(original, config):
    rotated = np.array([rotate([0, 0], point, config[2]) for point in original])
    center = [config[0], config[1]]
    return np.array([point + center for point in rotated])

def to_global(local_landmarks, global_pose):
    num_landmarks = int(np.size(local_landmarks) / 2)
    landmark_coords = np.zeros((num_landmarks, 2))
    for i in range(num_landmarks):
        landmark_coords[i][0] = global_pose[0] + local_landmarks[i*2] * np.cos(global_pose[2] + local_landmarks[(i*2)+1])
        landmark_coords[i][1] = global_pose[1] + local_landmarks[i*2] * np.sin(global_pose[2] + local_landmarks[(i*2)+1])
    return landmark_coords

def translational_error(pose1, pose2):
    deltax = pose2[0] - pose1[0]
    deltay = pose2[1] - pose1[1]
    return np.sqrt(deltax**2 + deltay**2)

def adjust_angle(angle):
    while np.abs(angle) > np.pi:
        if angle > np.pi:
            angle -= 2 * np.pi
        else:
            angle += 2 * np.pi
    return angle

def rotational_error(pose1, pose2):
    angle = adjust_angle(pose2[2] - pose1[2])
    return np.abs(angle)

parser = argparse.ArgumentParser()
parser.add_argument('--map', type=str, nargs='?')
parser.add_argument('--execution', type=str, nargs='?')
parser.add_argument('--estimates', type=str, nargs='?')

args = parser.parse_args()

landmarks = np.load(args.map, allow_pickle=True)

ground_truths = np.load(args.execution, allow_pickle=True)

guesses = np.load(args.estimates, allow_pickle=True)

readings = np.load('readings/readings_' + args.estimates[14] + '_' + args.estimates[16] + '_' + args.estimates[18] + '.npy', allow_pickle=True)

q = np.array(guesses[0])

fig, ax = plt.subplots(1, 3, figsize=(12,5))

ax[0].set_aspect('equal')
ax[0].set_xlim(0, 2)
ax[0].set_ylim(0, 2)
ax[0].set_title('guessed (black) vs ground truth (blue) motion')
ax[1].set_xlim(0, 20)
ax[1].set_ylim(0, 1)
ax[1].set_title('translational error')
ax[1].set_xlabel('time (seconds)')
ax[1].set_ylabel('distance from ground truth (meters)')
ax[2].set_xlim(0, 20)
ax[2].set_ylim(0, 0.5)
ax[2].set_title('rotational error')
ax[2].set_xlabel('time (seconds)')
ax[2].set_ylabel('angle from ground truth (radians)')



def init():
    global q
    q = np.array(guesses[0])

gt_patch = plt.Polygon(transform_rigid_body(original, ground_truths[0]), fill=False, color='blue')
guessed_patch = plt.Polygon(transform_rigid_body(original, guesses[0]), fill=False, color='black')

ax[0].add_patch(gt_patch)
ax[0].add_patch(guessed_patch)

ax[0].plot(landmarks[:,0], landmarks[:,1], 'o', markersize=5)

landmark_guesses, = ax[0].plot(landmarks[:,0], landmarks[:,1], 'x', markersize=5, color='red')

lines = []

translational_error_data = []
rotational_error_data = []

for i in range(np.size(landmarks, 0)):
    line, = ax[0].plot([landmarks[i][0], q[0]], [landmarks[i][1], q[1]], linewidth=0.5, color='black')
    lines.append(line)

def func(i, n):
    print(f'saving frame {i}/{n}')

def animate(i):
    global translational_error_data
    global rotational_error_data
    global q
    global landmark_guesses

    gt_patch.set_xy(transform_rigid_body(original, ground_truths[i+1]))

    guessed_patch.set_xy(transform_rigid_body(original, guesses[i]))

    ax[0].plot(ground_truths[i+1][0], ground_truths[i+1][1], 'o', color='blue', markersize=1.5)
    ax[0].plot(guesses[i][0], guesses[i][1], 'o', color='black', markersize=1.5)

    landmark_guesses.remove()

    sensed_landmarks = to_global(readings[(i*2)+2], guesses[i])

    for j in range(np.size(landmarks, 0)):
        lines[j].remove()
        line, = ax[0].plot([sensed_landmarks[j][0], q[0]], [sensed_landmarks[j][1], q[1]], linewidth=0.5, color='black')
        lines[j] = line
    
    landmark_guesses, = ax[0].plot(sensed_landmarks[:,0], sensed_landmarks[:,1], 'x', color='red', markersize=5)

    translational_error_data.append([i*.1, translational_error(guesses[i], ground_truths[i+1])])
    rotational_error_data.append([i*.1, rotational_error(guesses[i], ground_truths[i+1])])

    t_data = np.array(translational_error_data)
    r_data = np.array(rotational_error_data)

    ax[1].plot(t_data[:,0], t_data[:,1], color='green', linewidth=0.5)
    ax[2].plot(r_data[:,0], r_data[:,1], color='green', linewidth=0.5)

ani = FuncAnimation(fig, animate, interval=40, frames=200, init_func=init, repeat=False)
plt.show()
#plt.close()
ani.save('eval2/eval2_0_0_L_200.mp4', writer='ffmpeg')