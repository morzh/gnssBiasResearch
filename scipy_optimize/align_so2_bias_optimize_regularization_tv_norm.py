import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import sophus
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import cvxpy as cp
from cvxpy_atoms.total_variation_1d import *

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
bias_accum = np.zeros((d, N))
z = np.zeros((2*N,))
w = np.zeros((2*N,))


alpha = 0.01

y = R @ (x + b_gt)

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

def func_optimize(params, x, y, z, w, rho):
    theta = params[0]
    R = Rotation.from_euler('z', theta, degrees=False)
    R = R.as_matrix()[0:2, 0:2]
    bias = np.zeros((d, N))
    bias[0] = params[1:N+1]
    bias[1] = params[N+1:2*N+1]
    e = y - (R @ x + bias)
    e1 = bias[0] - z[0] + w[0] / rho
    e2 = bias[1] - z[1] + w[1] / rho
    return np.sum(e.flatten()**2) + 0.5*rho * np.sum(e1.flatten()**2) + 0.5*rho * np.sum(e2.flatten()**2)


def func_init_guess(params, x, y):
    theta = params[0]
    R = Rotation.from_euler('z', theta, degrees=False)
    R = R.as_matrix()[0:2, 0:2]
    e = y - R @ x
    return np.sum(e.flatten() ** 2)


init_guess = np.zeros((1,))
optimize_init_guess = scipy.optimize.minimize(func_init_guess, init_guess, args=(x, y))


rho = 1.5
lmbda = 2.0
init_guess = np.zeros((2*N+1,))
init_guess[0] = optimize_init_guess.x_1[0]
print('init guess is:', init_guess[0])
for idx in range(30):
    # ||R(\theta) x - y ||_2^2
    res_optimize = scipy.optimize.minimize(func_optimize, init_guess, args=(x, y, z, w, rho), options={'maxiter': 8})
    print('iteration', str(idx), 'theta is:', res_optimize.x_1[0])
    # |\nabla z| + ||b-z+w||**2
    theta__ = res_optimize.x_1[0]
    bias__ = res_optimize.x_1[1:]

    w_reg = w.reshape(2, -1)
    b_reg = res_optimize.x_1[1:].reshape(2, -1)
    z_reg = cp.Variable((2, N))
    funcs = [0.5 * rho * cp.sum_squares(b_reg - z_reg - w_reg/rho), lmbda * tv_norm_rows(z_reg)]
    prob = cp.Problem(cp.Minimize(sum(funcs)))
    result = prob.solve()
    z = (z_reg.value).reshape(-1,)
    w += rho*(bias__ - z)
    init_guess = res_optimize.x_1

    ##===========================================================##
    theta_optimize = res_optimize.x_1[0]
    bias_optimize = res_optimize.x_1[1:].reshape(2, -1)

    R_optimize = Rotation.from_euler('z', theta_optimize, degrees=False)
    R_optimize = R_optimize.as_matrix()[0:2, 0:2]
    x_test_optimize = R_optimize @ x + bias_optimize

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(x_test_optimize[0], x_test_optimize[1], linewidth=5)
    plt.plot(y[0], y[1])
    plt.subplot(1, 2, 2)
    plt.plot(bias_optimize[0], bias_optimize[1])
    plt.scatter(bias_optimize[0], bias_optimize[1])
    plt.tight_layout()
    plt.show()

# print(res2)
##===================================================================================##

theta_optimize = res_optimize.theta_hat[0]
bias_optimize = res_optimize.theta_hat[1:].reshape(2, -1)

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
plt.scatter(bias_optimize[0], bias_optimize[1])
plt.tight_layout()
plt.show()


