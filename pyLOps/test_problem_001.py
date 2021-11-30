import numpy as np
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation
import pylops
import matplotlib.pyplot as plt

N = 10
theta = 0.6
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

x = np.random.rand(2, N)
y = np.matmul(R, x)
r = np.eye(2)
Rop = pylops.LinearOperator(r)

xinv = pylops.optimization.leastsquares.RegularizedInversion(Rop, [], y, **dict(damp=0,  iter_lim=10, show=1))


print(xinv)

plt.scatter(x[0], x[1])
plt.scatter(y[0], y[1])
plt.show()




