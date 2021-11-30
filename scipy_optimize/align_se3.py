import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import sophus

N = 100
t = np.array([1.2, 2.0, 1.5])
r = np.array([0.6, -0.2, 1.0])
T = sophus.SE3.exp(np.hstack((t, r)))
T = T.matrix()
print(T)

x = np.random.rand(3, N)
y = np.matmul(T[0:3, 0:3], x) + T[0:3, 3].reshape(3, 1)


def fun(se3_params, x, y):
    T = sophus.SE3.exp(se3_params).matrix()
    e = y - np.matmul(T[0:3, 0:3], x) + T[0:3, 3].reshape(3, 1)
    return np.sum(e.flatten()**2)


def func(se3_params, x, y):
    T = sophus.SE3.exp(se3_params).matrix()
    e = y - np.matmul(T[0:3, 0:3], x) + T[0:3, 3].reshape(3, 1)
    return e.flatten()

init_guess = np.zeros((6,))

res1 = scipy.optimize.minimize(fun, init_guess,  args=(x, y))
res2 = least_squares(func, init_guess,  args=(x, y))
print(res1.x_1)
print('--------------------------------------------------------------------')
print('--------------------------------------------------------------------')
print(res2.x)