from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

dims = 2
step_n = 10000
step_set = [-0.1, 0, 0.1]
step_shape = (step_n, dims)
origin = np.zeros((1,dims))# Simulate steps in 2Dstep_shape = (step_n,dims)
steps = np.random.choice(a=step_set, size=step_shape)
path = np.concatenate([origin, steps]).cumsum(0)
start = path[:1]
stop = path[-1:]# Plot the pathfig = plt.figure(figsize=(8,8),dpi=200)

fig = plt.figure(figsize=(8, 8), dpi=200)
ax = fig.add_subplot(111)
ax.scatter(path[:, 0], path[:, 1], c='blue', alpha=0.25, s=0.05)
ax.plot(path[:, 0], path[:, 1], c='blue', alpha=0.5, lw=0.25, ls='-')
ax.plot(start[:, 0], start[:, 1], c='red', marker=''+'')
ax.plot(stop[:, 0], stop[:, 1], c='black', marker='o')
plt.title('2D Random Walk')
plt.tight_layout(pad=0)
plt.show()