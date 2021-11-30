import numpy as np
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation
from  matplotlib import pyplot as plt

N = 100
theta = 0.65
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

x = np.random.rand(2, N)
b_gt = np.zeros((2, N))
# b_gt[1, int(0.5*N+15):] = 5.5
y = R @ (x + b_gt)

def model(theta, x):
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.matmul(rot, x)


def fun(theta, x, y):
    theta__ = float(theta)
    rot = np.array([[np.cos(theta__), -np.sin(theta__)], [np.sin(theta__), np.cos(theta__)]])
    e = y - np.matmul(rot, x)
    return e.flatten()

def func(x, theta):
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.matmul(rot, x)

res = least_squares(fun, 0.0,  args=(x, y), verbose=1)
print(res.x[0])


##==========================================================================================

R = Rotation.from_euler('z', res.x[0], degrees=False)
R = R.as_matrix()[0:2, 0:2]
y_check = R @ x

plt.scatter(y[0], y[1], s=35)
plt.scatter(y_check[0], y_check[1], s=10)
plt.show()