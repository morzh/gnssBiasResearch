import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import sophus
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

d = 2
N = 100
r = 0.65

R = Rotation.from_euler('z', r, degrees=False)
R = R.as_matrix()[0:2, 0:2]
print(R)

linspace = np.linspace(0.0, float(N-1), N)
x = np.zeros((d, N))
x = 0.2*np.random.randn(d, N)
x[0] = linspace

b_gt = np.zeros((d, N))
b_gt[1, int(0.5*N+15):] = 5.5

bias_optimize = np.zeros((d, N))
bias_least_squares = np.zeros((d, N))
y = R @ (x + b_gt) + 0.2*np.random.randn(d, N)

plt.plot(x[0], x[1])
plt.plot(y[0], y[1])
plt.show()

# plt.stem(b_gt[0], b_gt[1])
# plt.show()


def func_least_squares(params, x, y):
    theta = params[0]
    R = Rotation.from_euler('z', theta, degrees=False)
    R = R.as_matrix()[0:2, 0:2]
    bias = np.zeros((d, N))
    bias[0] = params[1:N+1]
    bias[1] = params[N+1:2*N+1]
    e = y - (R @ x + bias)
    return e.flatten()


def func_optimize(params, x, y):
    theta = params[0]
    R = Rotation.from_euler('z', theta, degrees=False)
    R = R.as_matrix()[0:2, 0:2]
    bias = np.zeros((d, N))
    bias[0] = params[1:N+1]
    bias[1] = params[N+1:2*N+1]
    e = y - (R @ x + bias)
    return np.sum(e.flatten()**2)

init_guess = np.zeros((2*N+1,))

res_least_squares = least_squares(func_least_squares, init_guess, args=(x, y))
res_optimize = scipy.optimize.minimize(func_optimize, init_guess, args=(x, y))
# print(res_least_squares)
# print(res_optimize)
##===================================================================================##
# theta = 0.65
theta_least_squares = res_least_squares.x[0]
bias_least_squares[0] = res_least_squares.x[1:N + 1]
bias_least_squares[1] = res_least_squares.x[N + 1:2 * N + 1]

R_least_squares = Rotation.from_euler('z', theta_least_squares, degrees=False)
R_least_squares = R_least_squares.as_matrix()[0:2, 0:2]
x_test_least_squares = R_least_squares @ x + bias_least_squares
##===================================================================================##

theta_optimize = res_optimize.x_1[0]
bias_optimize[0] = res_optimize.x_1[1:N + 1]
bias_optimize[1] = res_optimize.x_1[N + 1:2 * N + 1]

R_optimize = Rotation.from_euler('z', theta_optimize, degrees=False)
R_optimize = R_optimize.as_matrix()[0:2, 0:2]
x_test_optimize = R_optimize @ x + bias_optimize
##===================================================================================##

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(x_test_least_squares[0], x_test_least_squares[1], linewidth=5)
plt.plot(y[0], y[1])
plt.subplot(1, 2, 2)
plt.plot(x_test_optimize[0], x_test_optimize[1], linewidth=5)
plt.plot(y[0], y[1])
plt.tight_layout()
plt.show()


print(res_optimize.x_1[0])
print(res_least_squares.x[0])

# print(np.abs(res_least_squares.x - res_optimize.x))
# print(np.sum(np.abs(res_least_squares.x - res_optimize.x)))

