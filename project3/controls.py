import numpy as np
import argparse
import matplotlib.pyplot as plt
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

def out_of_bounds(q):
    return q[0] > 1.8 or q[0] < 0.2 or q[1] > 1.8 or q[1] < 0.2

def rotate(origin, point, angle):
    newpointx = origin[0] + np.cos(angle) * (point[0] - origin[0]) - np.sin(angle) * (point[1] - origin[1])
    newpointy = origin[1] + np.sin(angle) * (point[0] - origin[0]) + np.cos(angle) * (point[1] - origin[1])
    return [newpointx, newpointy]

def transform_rigid_body(original, config):
    rotated = np.array([rotate([0, 0], point, config[2]) for point in original])
    center = [config[0], config[1]]
    return np.array([point + center for point in rotated])

fig, ax = plt.subplots(figsize=(12,5), nrows=2, ncols=5)

for x in range(5):
    map = np.load('maps/landmark_' + str(x) + '.npy', allow_pickle=True)
    for y in range(2):

        ax[y][x].set_aspect('equal')
        ax[y][x].set_xlim(0, 2)
        ax[y][x].set_ylim(0, 2)
        ax[y][x].plot(map[:,0], map[:,1], 'o', markersize=1.5, color='red')

        while(True):
            qstart = [np.random.rand() * 2, np.random.rand() * 2, (np.random.rand() * 2 * np.pi) - np.pi]
            if not out_of_bounds(qstart):
                break

        initial_patch = plt.Polygon(transform_rigid_body(original, qstart), fill=False, color='blue')
        ax[y][x].add_patch(initial_patch)

        q = [qstart[0], qstart[1], qstart[2]]
        control_sequence = np.zeros((201, 3))
        control_sequence[0] = [qstart[0], qstart[1], qstart[2]]
        index_counter = 1

        while(True):
            if index_counter == 201:
                break

            u = [np.random.rand() * 0.5, (np.random.rand() * 1.8) - 0.9]

            qtemp = [q[0], q[1], q[2]]

            oob = False
            for i in range(20):
                qtemp += drive_model(qtemp, u)
                if out_of_bounds(qtemp):
                    oob = True

            if not oob:
                control_sequence[index_counter:index_counter+20] = [u[0], u[1], 0]
                index_counter += 20
                q = qtemp
        
        positions = np.zeros((200, 2))

        for i in range(200):
            qstart += drive_model(qstart, control_sequence[i+1])
            positions[i] = [qstart[0], qstart[1]]
        
        ax[y][x].plot(positions[:,0], positions[:,1], markersize=1.5)
        np.save('controls/controls_' + str(x) + '_' + str(y), control_sequence)

plt.savefig('control_sequences_visualization.png')
plt.show()
