import numpy as np
import matplotlib.pyplot as plt

k = 10
m = 2

points = np.random.random((m, k))
centers = np.mean(points, axis=1).reshape(m, -1)

points__ = points - centers
points__norms = np.linalg.norm(points__, axis=1).reshape(m, 1)
points__ /= points__norms

print(points)
print(points__)

plt.figure(figsize=(20, 10))
# plt.scatter(diffs_points_1[0], diffs_points_1[1])
plt.scatter(points__[0], points__[1])
plt.show()