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
# print(R)

linspace = np.linspace(0.0, float(N-1), N)
x = np.zeros((d, N))
x[0] = linspace

b_gt = np.zeros((d, N))
b_gt[1, int(0.5*N):] = 5.5

bias_optimize = np.zeros((d, N))
z = np.zeros((d, N))
w = np.zeros((d, N))

rho = 0.01
lmbda = 0.01
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
for idx in range(10):
    res_optimize = scipy.optimize.minimize(func_optimize, init_guess, args=(x, y), options={'maxiter': 1})
    print('iteration', str(idx), 'theta is:', res_optimize.theta_hat[0])
    init_guess[0] = res_optimize.theta_hat[0]
    grad = res_optimize.jac[1:]
    b = proxL1Norm(res_optimize.theta_hat[1:] - alpha * grad, alpha * lmbda)
    init_guess[1:] = b.flatten()
# print(res2)
##===================================================================================##


theta_optimize = res_optimize.theta_hat[0]
bias_optimize[0] = res_optimize.theta_hat[1:N + 1]
bias_optimize[1] = res_optimize.theta_hat[N + 1:2 * N + 1]

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
plt.plot(linspace+bias_optimize[0], bias_optimize[1])
plt.tight_layout()
plt.show()


