import matplotlib.pyplot as plt
from KF.generate_trajectory import *

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def get_rotation_2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


R = get_rotation_2d(1.0)
points_1 = 2.0*5.0*generate_trajectory(num_samples=200, noise_mult=0.003)
points_2 = np.matmul(R, points_1) + 5*np.random.random((2, 1))

plt.figure(figsize=(20, 10))
plt.scatter(points_1[0], points_1[1])
plt.scatter(points_2[0], points_2[1])
plt.show()


diffs_points_1 = points_1[:, 1:] - points_1[:, :-1]
diffs_points_2 = points_2[:, 1:] - points_2[:, :-1]
plt.figure(figsize=(20, 10))
plt.plot(diffs_points_1[0], diffs_points_1[1])
plt.plot(diffs_points_2[0], diffs_points_2[1])
plt.show()


diffs_points_2_reverted = np.matmul(R.T, diffs_points_2)
plt.figure(figsize=(20, 10))
plt.scatter(diffs_points_1[0], diffs_points_1[1])
plt.scatter(diffs_points_2_reverted[0], diffs_points_2_reverted[1])
plt.show()