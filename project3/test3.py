import numpy as np
import matplotlib.pyplot as plt
import os

# Define the paths and parameters
landmark_maps = ["maps/landmark_0.npy", "maps/landmark_1.npy", "maps/landmark_2.npy", "maps/landmark_3.npy", "maps/landmark_4.npy"]
first_numbers = range(5)
second_numbers = range(2)
third_letters = ["H", "L"]
noise_levels = [200, 1000]

# Iterate through all combinations
for first_num in first_numbers:
    for second_num in second_numbers:
        for third_letter in third_letters:
            for noise_level in noise_levels:
                # Generate control sequence filename
                reading_filename = f"readings/readings_{first_num}_{second_num}_{third_letter}.npy"

                # Generate a unique output filename based on the combination
                output_filename = f"estim1/estm1_{first_num}_{second_num}_{third_letter}_{noise_level}.npy"

                # Run the particle filter for the current combination
                os.system(f"python particle_filter.py --map {landmark_maps[first_num]} --sensing {reading_filename} --num_particles {noise_level} --estimates {output_filename}")

