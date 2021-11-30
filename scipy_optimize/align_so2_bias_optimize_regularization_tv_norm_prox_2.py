import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import sophuspy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import cvxpy as cp
from cvxpy_atoms.total_variation_1d import *

d = 2
N = 100
r = 0.65
noise_mult = 0.1

R = Rotation.from_euler('z', r, degrees=False)
R = R.as_dcm()[0:2, 0:2]
print(R)

linspace = np.linspace(0.0, float(N-1), N)
x = np.zeros((d, N))
x[0] = linspace

b_gt = np.zeros((d, N))
b_gt[1, int(0.5*N):int(0.5*N+35)] = 5.5

bias_optimize = np.zeros((d, N))
bias_accum = np.zeros((d, N))
z = np.zeros((2, N))
z__ = np.zeros((2, N))
w = np.zeros((2, N))


alpha = 0.01

y = R @ x + b_gt + noise_mult * np.random.randn(d, N)
x += noise_mult * np.random.randn(d, N)

plt.plot(x[0], x[1])
plt.plot(y[0], y[1])
plt.show()

# plt.stem(b_gt[0], b_gt[1])
# plt.show()
def clip(beta, alpha):
    clipped = np.minimum(beta, alpha)
    clipped = np.maximum(clipped, -alpha)
    return clipped

def proxL1Norm(betaHat, alpha, penalizeAll=True):
    out = betaHat - clip(betaHat, alpha)
    if not penalizeAll:
        out[0] = betaHat[0]
    return out

def func_optimize(params, x, y):
    theta = params[0]
    R = Rotation.from_euler('z', theta, degrees=False)
    R = R.as_dcm()[0:2, 0:2]
    bias = params[1:].reshape(2, -1)
    e = y - (R @ x + bias)
    return np.sum(e.flatten()**2)

angle = 0.0
lmbda_2 = 0.05

init_guess = np.zeros((2*N+1,))
z_reg = cp.Variable((2, N))
for idx in range(200):
    # ||R(\theta) x - y ||_2^2
    res_optimize = scipy.optimize.minimize(func_optimize, init_guess, args=(x, y,), options={'maxiter': 5})
    print('iteration', str(idx), 'theta is:', res_optimize.x_1[0])
    theta__ = res_optimize.x_1[0]

    b_reg = res_optimize.x_1[1:].reshape(2, -1)
    grad = res_optimize.jac[1:].reshape(2, -1)
    step = 1.0
    b_reg = proxL1Norm(b_reg - step*grad, step*lmbda_2)

    init_guess = res_optimize.x_1.copy()
    init_guess[1:] = b_reg.flatten()

    if not idx % 30:
        theta_optimize = res_optimize.x_1[0]
        bias_optimize = res_optimize.x_1[1:].reshape(2, -1)

        R_optimize = Rotation.from_euler('z', theta_optimize, degrees=False)
        R_optimize = R_optimize.as_dcm()[0:2, 0:2]
        y_test = R_optimize @ x + bias_optimize

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.plot(y_test[0], y_test[1], linewidth=5)
        plt.plot(y[0], y[1])
        plt.subplot(1, 2, 2)
        plt.plot(bias_optimize[0], bias_optimize[1])
        plt.scatter(bias_optimize[0, 0], bias_optimize[1, 0], s=60, c='r')
        plt.scatter(bias_optimize[0], bias_optimize[1], s=20)
        plt.tight_layout()
        plt.show()

# print(res2)
##===================================================================================##

theta_optimize = res_optimize.x_1[0]
bias_optimize = res_optimize.x_1[1:].reshape(2, -1)

R_optimize = Rotation.from_euler('z', theta_optimize, degrees=False)
R_optimize = R_optimize.as_dcm()[0:2, 0:2]
y_test = R_optimize @ x + bias_optimize

print(theta_optimize)
print(bias_optimize[0])
print(bias_optimize[1])
##===================================================================================##

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(y_test[0], y_test[1], linewidth=5)
plt.plot(y[0], y[1])
plt.subplot(1, 2, 2)
plt.plot(bias_optimize[0], bias_optimize[1])
plt.scatter(bias_optimize[0], bias_optimize[1])
plt.tight_layout()
plt.show()


