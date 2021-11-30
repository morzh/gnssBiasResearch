import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import sophus
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import cvxpy as cp

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

def func_optimize(params, x, y):
    theta = params[0]
    R = Rotation.from_euler('z', theta, degrees=False)
    R = R.as_matrix()[0:2, 0:2]
    e = y - (R @ x)
    return np.sum(e.flatten()**2)

angle = 0.0
rho = 0.25
lmbda = 0.95
init_guess = np.zeros((1,))
for idx in range(10):
    # ||R(\theta) x - y ||_2^2
    res_optimize = scipy.optimize.minimize(func_optimize, init_guess, args=(x, y), options={'maxiter': 8})
    print('iteration', str(idx), 'theta is:', res_optimize.theta_hat[0])
    # |\nabla z| + ||b-z+w||**2
    theta__ = res_optimize.theta_hat[0]

    R__ = Rotation.from_euler('z', theta__, degrees=False)
    R__ = R__.as_matrix()[0:2, 0:2]
    x = R__ @ x

    z__ = cp.Variable(z.shape, value=z)
    z__.value = z
    funcs__ = [cp.sum_squares(x.flatten() - z__), lmbda * cp.tv(z__)]
    prob = cp.Problem(cp.Minimize(sum(funcs__)))
    result = prob.solve()
    z = z__.value

    angle += theta__
    init_guess = res_optimize.theta_hat
# print(res2)
##===================================================================================##

theta_optimize = res_optimize.theta_hat[0]
bias_optimize = z.reshape(2, -1)

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


