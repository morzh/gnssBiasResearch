import numpy as np
import matplotlib.pyplot as plt
from KF.KF_utils import *

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

"""
M. B. IGNAGNI
Separate-Bias Kalman Estimator with Bias State Noise
dynamics is:
x_k = A_{k-1} x_{k-1}  + B_{k-1} b_{k-1} + \\xi_{k-1}
b_{k} = b_{k-1} + \\betta_{k}

y_k = H_k x_k + C_k b_k + \eta_{k}
"""

def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

state_sz = 4
true_position_array = np.empty((2, 0))
estimated_state_array = np.empty((2*state_sz, 0))
state_covariance_bias_array = np.empty((2, 0))


dt = 0.1 # time step of mobile movement
x = np.zeros((state_sz, 1))
x[2:state_sz] = 0.1
b = np.zeros((state_sz, 1))
# b = 0.1*np.ones((2, 1))
P_x = 1.0 * np.eye(state_sz)
P_b = 1.2 * np.eye(state_sz)

A = np.eye(state_sz)
# B = 0.125*np.eye(state_sz)
B = np.zeros((state_sz, state_sz))
Q_x = 0.25 * np.eye(state_sz)
Q_b = 1.0 * np.eye(state_sz)

y = np.zeros((2, 1))
H = np.eye(2, state_sz)
C = np.eye(2, state_sz)
# C = np.zeros((2, state_sz))
R = 0.1*np.eye(2)


z = np.vstack((x, b))
F = np.vstack((np.hstack((A, B)), np.hstack((np.zeros(A.shape), np.eye(B.shape[0])))))
Q = np.eye((state_sz * 2))
Q[0:state_sz, 0:state_sz] = Q_x
Q[state_sz:2 * state_sz, state_sz:2 * state_sz] = Q_b
L = np.hstack((H, C))
P_tilde = np.vstack((np.hstack((P_x, np.zeros((P_x.shape[0], b.shape[0])))), np.hstack((np.zeros((b.shape[0], P_x.shape[0])), P_b))))
I = np.eye(state_sz*2)

# Simulation


N_iter = 500
# ys = get_trajectory(N_iter)

noise_mult = 0.015
bias = np.array([0.15, 0.3])
y = z[0:2, 0].reshape(2, 1) + noise_mult*abs(np.random.randn(2)).reshape(2, 1) + bias.reshape(2, 1)
# y = z[0:2, 0] + noise_mult*abs(np.random.randn(2)) + bias
y = y.reshape(2, 1)


print(z.flatten())
# print(P_tilde)
print('===========================================================')

for i in np.arange(0, N_iter):
    (z, P_tilde) = kf_predict(z, P_tilde, F, Q, np.eye(2*state_sz), np.zeros((2*state_sz, 1)))
    (z, P_tilde, _, _, _, _) = kf_update(z, P_tilde, y, L, R)

    estimated_state_array = np.hstack((estimated_state_array, z))
    true_position_array = np.hstack((true_position_array, y))
    state_covariance_bias_array = np.hstack((state_covariance_bias_array, np.array([[P_tilde[4, 4]], [P_tilde[5, 5]]])))
    noise = noise_mult*abs(np.random.randn(2))
    y = z[0:2].reshape(2, 1) + noise.reshape(2, 1) + bias.reshape(2, 1)
    # y = z[0:2] + noise.reshape(2, 1) + bias.reshape(2, 1)
    # plt.plot(estimated_state_array[0], estimated_state_array[1])
    # plt.show()
    # print(noise)
    print(z.flatten())
    # print(P_tilde)
    print('===========================================================')


bias_x_mean = np.mean(estimated_state_array[4])
bias_x_std = np.std(estimated_state_array[4])
bias_y_mean = np.mean(estimated_state_array[5])
bias_y_std = np.std(estimated_state_array[5])
bias_x_median = np.median(estimated_state_array[4])
bias_x_mad = mad(estimated_state_array[4])
bias_y_median = np.median(estimated_state_array[5])
bias_y_mad = mad(estimated_state_array[5])

ts = np.arange(0.0, estimated_state_array.shape[1])
plt.figure(figsize=(20, 10))
plt.plot(estimated_state_array[0], estimated_state_array[1])
plt.plot(true_position_array[0], true_position_array[1])
plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
# ax1 = plt.axes()
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
filter_equations = '$x_n =  A_n x_{n-1} + B_n b_{n-1} + \\xi_n$ \n'
filter_equations += '$b_n = b_{n-1} + \\beta_n$ \n'
filter_equations += '$y_n = H_n x_n + C_n b_n  + \\eta_n$ \n'
filter_equations += 'bias ground true: [' + str(bias[0]) + ' ' + str(bias[1]) + '] \n'
filter_equations += 'bias estimated mean/std: [' + str(np.round(bias_x_mean, 4)) + ' ' + str(np.round(bias_y_mean, 4)) + ']/[' + str(np.round(bias_x_std, 4)) + ' ' + str(np.round(bias_y_std, 4)) +']\n'
filter_equations += 'bias estimated median/MAD: [' + str(np.round(bias_x_median, 4)) + ' ' + str(np.round(bias_y_median, 4)) + ']/[' + str(np.round(bias_x_mad, 4)) + ' ' + str(np.round(bias_y_mad, 4)) +']\n'
plt.text(0.20, np.max(estimated_state_array[4]), filter_equations, fontsize=12, verticalalignment='top', bbox=props)
plt.plot(ts, estimated_state_array[4])
# plt.errorbar(ts, estimated_state_array[4], yerr=state_covariance_bias_array[0])
# plt.plot(ts, estimated_state_array[4]-state_covariance_bias_array[0], color='black')
# plt.plot(ts, estimated_state_array[4]+state_covariance_bias_array[0], color='black')
plt.hlines(bias[0], ts[0], ts[-1], colors='red', linewidth=5)
plt.subplot(2, 1, 2)
plt.plot(ts, estimated_state_array[5])
# plt.plot(ts, estimated_state_array[5] - state_covariance_bias_array[1])
plt.hlines(bias[1], ts[0], ts[-1], colors='red', linewidth=5)
plt.show()

