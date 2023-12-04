import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys

dt = 0.1
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

parser = argparse.ArgumentParser()
parser.add_argument('--map', type=str, nargs='?')
parser.add_argument('--execution', type=str, nargs='?')
parser.add_argument('--sensing', type=str, nargs='?')

args = parser.parse_args()

landmarks = np.load(args.map, allow_pickle=True)

ground_truths = np.load(args.execution, allow_pickle=True)

readings = np.load(args.sensing, allow_pickle=True)

q = np.array(readings[0])

fig, ax = plt.subplots(dpi=100)

ax.set_aspect('equal')
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)


def init():
    global q
    q = np.array(readings[0])

gt_patch = plt.Polygon(transform_rigid_body(original, ground_truths[0]), fill=False, color='blue')
sensed_patch = plt.Polygon(transform_rigid_body(original, readings[0]), fill=False, color='red')

ax.add_patch(gt_patch)
ax.add_patch(sensed_patch)

plt.plot(landmarks[:,0], landmarks[:,1], 'o', markersize=5)

landmark_guesses, = plt.plot(landmarks[:,0], landmarks[:,1], 'x', markersize=5)

lines = []

for i in range(np.size(landmarks, 0)):
    line, = plt.plot([landmarks[i][0], q[0]], [landmarks[i][1], q[1]], linewidth=0.5, color='red')
    lines.append(line)


def animate(i):
    print(i)
    global q
    global landmark_guesses

    gt_patch.set_xy(transform_rigid_body(original, ground_truths[i+1]))

    q += drive_model(q, readings[(2*i)+1])
    sensed_patch.set_xy(transform_rigid_body(original, q))

    ax.plot(ground_truths[i][0], ground_truths[i][1], 'o', color='blue', markersize=1.5)
    ax.plot(q[0], q[1], 'o', color='red', markersize=1.5)

    landmark_guesses.remove()

    sensed_landmarks = to_global(readings[(i*2)+2], q)

    for j in range(np.size(landmarks, 0)):
        lines[j].remove()
        line, = plt.plot([sensed_landmarks[j][0], q[0]], [sensed_landmarks[j][1], q[1]], linewidth=0.5, color='red')
        lines[j] = line
    
    landmark_guesses, = ax.plot(sensed_landmarks[:,0], sensed_landmarks[:,1], 'x', color='red', markersize=5)


ani = FuncAnimation(fig, animate, interval=70, frames=200, init_func=init, repeat=False)

plt.show()
plt.close()
