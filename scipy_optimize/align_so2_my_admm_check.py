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


np.random.seed(19)
N = 40
d = 200
nnz = 3

prm = np.random.permutation(d+1)
betaTrue = np.zeros(d+1)
betaTrue[prm[0:nnz]] = 5*np.random.randn(nnz)
z = np.zeros(betaTrue.shape)
w = np.zeros(betaTrue.shape)

X = np.random.randn(N, d)
X = np.insert(X, 0, 1, axis=1)
noise = 0.001 * np.random.randn(N)
y = X @ betaTrue + noise

plt.stem(betaTrue)
plt.title('Beta ground truth')
plt.show()


def func_optimize(beta, X, y, gamma, w, rho):
    e = X @ beta - y
    e2 = beta - gamma + w
    return 0.5*np.sum(e**2) + 0.5*rho * np.sum(e2**2)


lmbda = 3
rho = 1.0
maxIter = 200

gamma = cp.Variable(betaTrue.shape)
gamma.value = np.zeros(betaTrue.shape)
init_guess = np.zeros(betaTrue.shape)
costFunVals = np.zeros(maxIter)

for idx in range(maxIter):
    print('iteration', str(idx).zfill(3))
    least_squares = scipy.optimize.minimize(func_optimize, init_guess, args=(X, y, z, w, rho))
    beta = least_squares.x_1


    if not idx % 50:
        plt.figure()
        plt.title('beta optimization step')
        plt.stem(least_squares.x_1, markerfmt='C1o')
        plt.stem(betaTrue)
        plt.show()


    funcs = [0.5 * rho * cp.sum_squares(beta - gamma + w), lmbda * cp.norm1(gamma)]
    prob = cp.Problem(cp.Minimize(sum(funcs)))
    result = prob.solve(warm_start=True)

    if not idx % 50:
        plt.figure()
        plt.title('gamma optimization step')
        plt.stem(gamma.value, markerfmt='C1o')
        plt.stem(betaTrue)
        plt.show()


    w += (least_squares.x_1 - gamma.value) * rho
    init_guess = least_squares.x_1

    beta = least_squares.x_1
    costFunVals[idx] = 0.5 * np.linalg.norm(X @ beta - y) ** 2 + lmbda * np.sum(np.abs(beta))




plt.figure()
plt.stem(beta, markerfmt='C1o')
plt.stem(betaTrue)
plt.show()

plt.semilogy(costFunVals)
plt.show()

