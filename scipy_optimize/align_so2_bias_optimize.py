import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import sophuspy
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
x[0] = linspace

b_gt = np.zeros((d, N))
b_gt[1, int(0.5*N):] = 5.5

bias_optimize = np.zeros((d, N))
bias_least_squares = np.zeros((d, N))
y = R @ (x + b_gt)

plt.plot(x[0], x[1])
plt.plot(y[0], y[1])
plt.show()

# plt.stem(b_gt[0], b_gt[1])
# plt.show()
def func_optimize(params, x, y):
    theta = params[0]
    R = Rotation.from_euler('z', theta, degrees=False)
    R = R.as_matrix()[0:2, 0:2]
    bias = np.zeros((d, N))
    bias[0] = params[1:N+1]
    bias[1] = params[N+1:2*N+1]
    e = y - (R @ x + bias)
    return np.sum(e.flatten())


init_guess = np.zeros((2*N+1,))
for idx in range(50):
    res_optimize = scipy.optimize.minimize(func_optimize, init_guess, args=(x, y), options={'maxiter': 1})
    init_guess = res_optimize.x_1

# print(res2)
##===================================================================================##

theta_optimize = res_optimize.x_1[0]
bias_optimize[0] = res_optimize.x_1[1:N + 1]
bias_optimize[1] = res_optimize.x_1[N + 1:2 * N + 1]

R_optimize = Rotation.from_euler('z', theta_optimize, degrees=False)
R_optimize = R_optimize.as_matrix()[0:2, 0:2]
x_test_optimize = R_optimize @ x + bias_optimize

print(theta_optimize)
print(bias_optimize[0])
print(bias_optimize[1])
##===================================================================================##

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(x_test_optimize[0], x_test_optimize[1], linewidth=5)
plt.plot(y[0], y[1])
plt.subplot(1, 2, 2)
plt.plot(bias_optimize[0], bias_optimize[1])
plt.tight_layout()
plt.show()


