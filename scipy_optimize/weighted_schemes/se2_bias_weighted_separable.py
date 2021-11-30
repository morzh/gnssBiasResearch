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


def least_sqaures_se2(params, x, y, weights):
    theta = params[0]
    translate = params[1:3]
    R = Rotation.from_euler('z', theta, degrees=False)
    R = R.as_matrix()[0:2, 0:2]
    e = weights * (y - R @ x - translate.reshape(2, 1))
    return e.reshape((e.size, ))


def admm_f1(params, x, y, z, w, rho, weights):
    se2_params = params[0:3]
    theta = se2_params[0]
    translate = se2_params[1:3]
    R = Rotation.from_euler('z', theta, degrees=False)
    R = R.as_matrix()[0:2, 0:2]
    bias = params[3:].reshape(2, -1)
    e = weights * (y - R @ x + translate.reshape(2, 1) - bias)
    e2 = bias - z + w
    return np.sum(e.flatten()**2) + rho * np.sum(e2.flatten()**2)
    # return 0.5*np.sum(e.flatten()**2) + 0.5*rho * np.sum(np.linalg.norm(e2, axis=0)**2)




dimensions = 2
number_elements = 100
angle = 0.65
angle_init_guess = 0.5
noise_multiplier = 0.1
group_num_parameters = 3
bias_point = int(0.5 * number_elements) + 5


rotation_init_guess = Rotation.from_euler('z', angle_init_guess, degrees=False)
rotation_init_guess = rotation_init_guess.as_matrix()[0:2, 0:2]
R = Rotation.from_euler('z', angle, degrees=False)
R = R.as_matrix()[0:2, 0:2]
print(R)

x_range = np.linspace(0.0, float(number_elements - 1), number_elements)
x = np.zeros((dimensions, number_elements))
x[0] = x_range


b_gt = np.zeros((dimensions, number_elements))
weights = np.ones(number_elements)
bias_optimize = np.zeros((dimensions, number_elements))
z = np.zeros((2, number_elements))
w = np.zeros((2, number_elements))


b_gt[1, bias_point:] = 5.5
weights[bias_point:] *= 2

alpha = 0.01

y = R @ x + b_gt + noise_multiplier * np.random.randn(dimensions, number_elements)
x = rotation_init_guess @ x
x += noise_multiplier * np.random.randn(dimensions, number_elements)

plt.plot(x[0], x[1])
plt.plot(y[0], y[1])
plt.show()

init_guess = np.zeros((group_num_parameters,))
init_guess[0] = angle_init_guess

estimated_se3_parameters = scipy.optimize.least_squares(least_sqaures_se2, init_guess, args=(x, y, weights))

estimated_rotation_matrix = Rotation.from_euler('z', estimated_se3_parameters.x[0], degrees=False)
estimated_rotation_matrix = estimated_rotation_matrix.as_matrix()[0:2, 0:2]
estimated_translation = estimated_se3_parameters.x[1:3]

estimated_y = estimated_rotation_matrix @ x + estimated_translation.reshape((2,1))

plt.figure(figsize=(20, 10))
plt.subplot(1, 1, 1)
plt.plot(estimated_y[0], estimated_y[1], linewidth=5, color='red')
plt.plot(y[0], y[1])
plt.tight_layout()
plt.show()


