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


def clip(beta, alpha):
    clipped = np.minimum(beta, alpha)
    clipped = np.maximum(clipped, -alpha)
    return clipped

def proxL1Norm(betaHat, alpha, penalizeAll=True):
    out = betaHat - clip(betaHat, alpha)
    if not penalizeAll:
        out[0] = betaHat[0]
    return out

def func_optimize(params, x, y, z, w, rho):
    se2_params = params[0:3]
    theta = se2_params[0]
    translate = se2_params[1:3]
    R = Rotation.from_euler('z', theta, degrees=False)
    R = R.as_matrix()[0:2, 0:2]
    bias = params[3:].reshape(2, -1)
    e = y - (R @ x + translate.reshape(2, 1) + bias)
    e2 = bias - z + w
    return np.sum(e.flatten()**2) + rho * np.sum(e2.flatten()**2)
    # return 0.5*np.sum(e.flatten()**2) + 0.5*rho * np.sum(np.linalg.norm(e2, axis=0)**2)

d = 2
N = 100
r = 0.65
r_init_guess = 0.75
noise_mult = 0.1
group_num_parameters = 3


init_guess_R = Rotation.from_euler('z', r_init_guess, degrees=False)
init_guess_R = init_guess_R.as_matrix()[0:2, 0:2]
R = Rotation.from_euler('z', r, degrees=False)
R = R.as_matrix()[0:2, 0:2]
print(R)

linspace = np.linspace(0.0, float(N-1), N)
x = np.zeros((d, N))
x[0] = linspace


b_gt = np.zeros((d, N))
b_gt[1, int(0.25*N) - 2:] = 5.5

bias_optimize = np.zeros((d, N))
bias_accum = np.zeros((d, N))
z = np.zeros((2, N))
w = np.zeros((2, N))


alpha = 0.01

y = R @ x + b_gt + noise_mult * np.random.randn(d, N)
x = init_guess_R @ x
x += noise_mult * np.random.randn(d, N)

plt.plot(x[0], x[1])
plt.plot(y[0], y[1])
plt.show()

# plt.stem(b_gt[0], b_gt[1])
# plt.show()


angle = 0.0
rho = 1.0
lambda_tv = 0.9
lmbda_l1 = 0.01
lmbda_l2 = 0.001

init_guess = np.zeros((2*N+group_num_parameters,))
z_reg = cp.Variable((2, N))
for idx in range(900):
    # ||R(\theta) x - y ||_2^2
    res_optimize = scipy.optimize.minimize(func_optimize, init_guess, args=(x, y, z, w, rho), options={'maxiter': 15})
    print('iteration', str(idx), 'theta is:', res_optimize.x_1[0], 'translate is:', res_optimize.x_1[1:3])
    theta__ = res_optimize.x_1[0]
    bias__ = res_optimize.x_1[3:]

    b_reg = res_optimize.x_1[3:].reshape(2, -1)
    funcs = [0.5*rho * cp.sum_squares(b_reg - z_reg + w), lambda_tv * tv_norm_cols(z_reg), lmbda_l2 * cp.sum_squares(z_reg)]
    # funcs = [0.5 * rho * cp.sum_squares(b_reg - z_reg + w), lambda_tv * tv_norm_cols(z_reg), lmbda_l1 * cp.sum(cp.norm1(z_reg, axis=0))]
    prob = cp.Problem(cp.Minimize(sum(funcs)))
    result = prob.solve(warm_start=True)

    w += b_reg - z_reg.value
    init_guess = res_optimize.x_1
    z = z_reg.value

    if not idx % 60:
        plt.subplot(2, 2, 1)
        plt.plot(z_reg.value[0], z_reg.value[1])
        plt.subplot(2, 2, 2)
        plt.plot(b_reg[0], b_reg[1])
        plt.subplot(2, 2, 3)
        plt.plot(w[0], w[1])
        plt.show()
        ##===========================================================##
        theta_optimize = res_optimize.x_1[0]
        translate_optimize = res_optimize.x_1[1:3].reshape(2, 1)
        bias_optimize = res_optimize.x_1[3:].reshape(2, -1)

        R_optimize = Rotation.from_euler('z', theta_optimize, degrees=False)
        R_optimize = R_optimize.as_matrix()[0:2, 0:2]

        y_test = R_optimize @ x + translate_optimize + bias_optimize
        y_test_2 = R_optimize @ x + translate_optimize

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.plot(y_test_2[0], y_test_2[1], linewidth=5)
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
bias_optimize = res_optimize.x_1[3:].reshape(2, -1)

R_optimize = Rotation.from_euler('z', theta_optimize, degrees=False)
R_optimize = R_optimize.as_matrix()[0:2, 0:2]
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


