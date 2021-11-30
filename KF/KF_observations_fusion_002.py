import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from KF.KF_utils import *
from KF.generate_trajectory import *

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


"""
D. Willner, C. B. Chang,  and K. P. Dunn
KALMAN FILTER ALGORITHMS FOR A MULTI-SENSOR SYSTEM
dynamics is:
x_{k+1} = A_{k} x_{k} + \\xi_{k} ~ Q(k+1)
z_{k+1, i} =  H_i x_{k+1} + v_{k+1, i} ~ R_{k+1, i}
"""

points_1 = 2.0*get_trajectory(num_samples=500, noise_mult=0.005)
points_2 = 2.0*get_trajectory(num_samples=500, noise_mult=0.003)

points_1 = 5.0*generate_trajectory(num_samples=500, noise_mult=0.01)
points_2 = 5.0*generate_trajectory(num_samples=500, noise_mult=0.006)

plt.plot(points_1[0], points_1[1])
plt.plot(points_2[0], points_2[1])
# plt.plot(points_1_[0], points_1_[1])
# plt.plot(points_2_[0], points_2_[1])
plt.show()



state_sz = 4
observe_sz = 2
observed_position_1_array = np.empty((2, 0))
observed_position_2_array = np.empty((2, 0))
estimated_state_array = np.empty((state_sz, 0))
state_covariance_bias_array = np.empty((2, 0))


dt = 1# time step of mobile movement
x = np.zeros((state_sz, 1))
x[0: 2] = 0.5*(get_closest_point(x[0: 2], points_1) + get_closest_point(x[0: 2], points_2)).reshape(2, 1)
vel = 0.5*(get_closest_point_next(x[0:2], points_1).reshape(2, 1) + get_closest_point_next(x[0:2], points_2).reshape(2, 1)) - x[0: 2]
vel = dt * vel / np.linalg.norm(vel)
x[2:state_sz] = vel
P = 1.0 * np.eye(state_sz)

A = np.eye(state_sz)
A[0, 2] = dt
A[1, 3] = dt
Q = 0.1 * np.eye(state_sz)

# y_1 = np.zeros((observe_sz, 1))
# y_2 = np.zeros((observe_sz, 1))
H_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
H_2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
# H_1 = np.eye(2, state_sz)
# H_2 = np.eye(2, state_sz)
R_1 = 0.1*np.eye(observe_sz)
R_2 = 0.1*np.eye(observe_sz)

'''
==========================SIMULATION===================================================
'''
N_iter = 500
'''
noise_mult = 1.5
bias = np.array([0.05, 0.3])
y_1 = x[0:2, 0].reshape(2, 1) + noise_mult * abs(np.random.randn(2)).reshape(2, 1)
y_2 = x[0:2, 0].reshape(2, 1) + noise_mult * abs(np.random.randn(2)).reshape(2, 1)
'''
# y_1 = get_closest_point_next(x[0:2]+dt*x[2:4], points_1).reshape(2, 1)
# y_2 = get_closest_point_next(x[0:2]+dt*x[2:4], points_2).reshape(2, 1)


print('===========================================================')
for i in np.arange(0, N_iter):
    (x__, P) = kf_predict(x, P, A, Q, np.eye(state_sz), np.zeros((state_sz, 1)))

    R = np.vstack((np.hstack((R_1, np.zeros(R_1.shape))), np.hstack((np.zeros(R_2.shape), R_2))))
    P_inv = np.linalg.inv(P) + np.matmul(H_1.T, np.matmul(np.linalg.inv(R_1), H_1)) + np.matmul(H_2.T, np.matmul(np.linalg.inv(R_2), H_2))
    P = np.linalg.inv(P_inv)
    K_1 = np.matmul(P, np.matmul(H_1.T, np.linalg.inv(R_1)))
    K_2 = np.matmul(P, np.matmul(H_2.T, np.linalg.inv(R_2)))

    y_1 = get_closest_point_iterative(x[0:2], x[2:4], dt, points_1).reshape(2, 1)
    y_2 = get_closest_point_iterative(x[0:2], x[2:4], dt, points_2).reshape(2, 1)
    # y_1 = get_closest_point(x__[0:2], points_1).reshape(2, 1)
    # y_2 = get_closest_point(x__[0:2], points_2).reshape(2, 1)

    estimated_state_array = np.hstack((estimated_state_array, x))
    observed_position_1_array = np.hstack((observed_position_1_array, y_1))
    observed_position_2_array = np.hstack((observed_position_2_array, y_2))

    plt.plot(points_1[0], points_1[1])
    plt.plot(points_2[0], points_2[1])
    plt.plot(estimated_state_array[0], estimated_state_array[1], color='grey', linewidth=3)
    plt.plot(observed_position_1_array[0], observed_position_1_array[1], color='red')
    plt.plot(observed_position_2_array[0], observed_position_2_array[1], color='green')
    plt.arrow(x[0, 0], x[1, 0], x[2, 0], x[3, 0])
    plt.scatter(x__[0, 0], x__[1, 0], c='k', s=35)
    plt.scatter(x[0, 0], x[1, 0], c='grey', s=35)
    plt.scatter(y_1[0, 0], y_1[1, 0], c='r')
    plt.scatter(y_2[0, 0], y_2[1, 0], c='g')
    # plt.scatter(observed_position_1_array[0, -1], observed_position_1_array[0, -1], c='r')
    # plt.scatter(observed_position_2_array[0, -1], observed_position_2_array[0, -1], c='g')
    plt.show()

    x = x__ + np.matmul(K_1, (y_1 - np.matmul(H_1, x__))) + np.matmul(K_2, (y_2 - np.matmul(H_2, x__)))

    '''
    np.random.seed(2*i)
    noise_1 = noise_mult * abs(np.random.randn(2))
    np.random.seed(2*i+1)
    noise_2 = noise_mult * abs(np.random.randn(2))
    y_1 = x[0:2].reshape(2, 1) + noise_1.reshape(2, 1)
    y_2 = x[0:2].reshape(2, 1) + noise_2.reshape(2, 1)
    '''

    # plt.plot(estimated_state_array[0], estimated_state_array[1])
    # plt.show()
    print(x.flatten())
    print('===========================================================')


ts = np.arange(0.0, estimated_state_array.shape[1])
plt.figure(figsize=(20, 10))
plt.plot(estimated_state_array[0], estimated_state_array[1], linewidth=6, color='grey')
plt.plot(observed_position_1_array[0], observed_position_1_array[1])
plt.plot(observed_position_2_array[0], observed_position_2_array[1])
plt.show()
