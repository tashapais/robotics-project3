import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--landmarks', type=int, nargs='?')
parser.add_argument('--name', type=str, nargs='?')
args = parser.parse_args()

map = np.ones((args.landmarks, 2))

for i in range(args.landmarks):
    x = np.random.rand() * 2
    y = np.random.rand() * 2
    map[i] = [x, y]
    
np.save(args.name, map)

fig, ax = plt.subplots(dpi=100)
ax.set_aspect('equal')
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)

plt.plot(map[:,0], map[:,1], 'o')
plt.show()
