import matplotlib.pyplot as plt
import time

x = range(10)
y = range(10)

fig, ax = plt.subplots(figsize=(11,4.5), nrows=2, ncols=5)

for row in ax:
    for col in row:
        col.set_aspect('equal')
        col.set_xlim(0, 2)
        col.set_ylim(0, 2)
        col.plot(x, y)

plt.show()