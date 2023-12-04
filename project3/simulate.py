import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import sys

dt = 0.1

def drive_model(q, u):
    dq = np.zeros_like(q)
    dq[0] = u[0] * np.cos(q[2]) * dt
    dq[1] = u[0] * np.sin(q[2]) * dt
    dq[2] = u[1] * dt
    return dq

def adjust_angle(angle):
    while np.abs(angle) > np.pi:
        if angle > np.pi:
            angle -= 2 * np.pi
        else:
            angle += 2 * np.pi
    return angle

def landmark_sensor(ground_truth_x, ground_truth_y, ground_truth_theta, landmarks):
    N = np.size(landmarks, 0)
    observations = np.zeros(N * 2)
    for i in range(N):
        deltax = landmarks[i][0] - ground_truth_x
        deltay = landmarks[i][1] - ground_truth_y
        angle = np.arctan2(deltay, deltax)
        local_angle = adjust_angle(angle - ground_truth_theta)
        distance = np.sqrt((deltax ** 2) + (deltay ** 2))
        observations[2*i] = distance
        observations[(2*i)+1] = local_angle
    return observations

parser = argparse.ArgumentParser()
parser.add_argument('--plan', type=str, nargs='?')
parser.add_argument('--map', type=str, nargs='?')
parser.add_argument('--execution', type=str, nargs='?')
parser.add_argument('--sensing', type=str, nargs='?')
args = parser.parse_args()

control_sequence = np.load(args.plan, allow_pickle=True)
landmark_map = np.load(args.map, allow_pickle=True)

landmarks_amount = np.size(landmark_map, 0)

executed_control_sequence = np.zeros((200, 2))

for i in range(200):
    planned_control = control_sequence[i+1]

    linear_noise = np.random.normal(0, 0.075)
    angular_noise = np.random.normal(0, 0.2)

    if planned_control[0] == 0:
        linear_noise = 0
    if planned_control[1] == 0:
        angular_noise = 0

    linear_control = np.clip(planned_control[0] + linear_noise, -0.5, 0.5)
    angular_control = np.clip(planned_control[1] + angular_noise, -0.9, 0.9)

    executed_control_sequence[i] = [linear_control, angular_control]

ground_truth = np.zeros((201, 3))
ground_truth[0] = control_sequence[0]

q = [ground_truth[0][0], ground_truth[0][1], ground_truth[0][2]]

for i in range(200):
    q += drive_model(q, executed_control_sequence[i])
    ground_truth[i+1] = q

np.save(args.execution, ground_truth)

sensed_control_sequence = np.zeros((200,2))

for i in range(200):
    high = 'H' in args.sensing

    executed_control = executed_control_sequence[i]

    linear_noise = 0
    angular_noise = 0

    if control_sequence[i+1][0] != 0 and high == True:
        linear_noise = np.random.normal(0, 0.1)
    if control_sequence[i+1][0] != 0 and high == False:
        linear_noise = np.random.normal(0, 0.05)
    if control_sequence[i+1][1] != 0 and high == True:
        angular_noise = np.random.normal(0, 0.3)
    if control_sequence[i+1][1] != 0 and high == False:
        angular_noise = np.random.normal(0, 0.1)

    linear_control = np.clip(executed_control[0] + linear_noise, -0.5, 0.5)
    angular_control = np.clip(executed_control[1] + angular_noise, -0.9, 0.9)

    sensed_control_sequence[i] = [linear_control, angular_control]

readings = np.zeros((401,landmarks_amount*2))
readings[0][0] = control_sequence[0][0]
readings[0][1] = control_sequence[0][1]
readings[0][2] = control_sequence[0][2]

for i in range(200):
    readings[(2*i)+1][0] = sensed_control_sequence[i][0]
    readings[(2*i)+1][1] = sensed_control_sequence[i][1]

    pose = ground_truth[i+1]
    observation_measurement = landmark_sensor(pose[0], pose[1], pose[2], landmark_map)

    for j in range(landmarks_amount):
        observation_measurement[2*j] += np.random.normal(0, 0.02)
        observation_measurement[2*j+1] += np.random.normal(0, 0.02)

    readings[(2*i)+2] = observation_measurement

np.save(args.sensing, readings)
