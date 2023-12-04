import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import sys

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

print(landmark_sensor(0, 0, np.pi, [[0, 1], [-2, 0], [0, -1]]))
