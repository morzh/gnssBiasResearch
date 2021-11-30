import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from KF.KF_utils import *
from KF.generate_trajectory import *

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

"""
dynamics is:
x_{k+1} = A_k x_k (+ B_k b_k) + \\xi_{k+1} ~ Q_x(k+1)
b_{k+1} = b_k + \\betta_{k+1} ~ Q_b(k+1)
s_{k+1} = s_k
z_{k+1, 1} =  H_{k, 1} x_{k+1} + C_k b_{k+1} + v_{k+1, 1} ~ R_{k+1, 1}
z_{k+1, 2} =  s_{k+1} H_{k, 2} x_{k+1} + v_{k+1, 2} ~ R_{k+1, 2}

x = [x^pos.T x^vel.T].T
A = [I_2 dt*I_2
     0_2  I_2]
H_1 = H_2 = [I 0]
"""
'''
centralized  KF is:
y = [x.T b.T s].T

y_{k+1} =  F_k y_k + \\eta_{k+1} ~ Q(k+1)
z_{k+1} = L_k y_{k+1} + v_{k+1} ~ R(k+1)

P = [P_11 P_12 0
     P_21 P_22 0
     0    0    0]
F = [A  (B) 0  
     O  I   0
     0  0   1]
Q = [Q_x 0   0
     0   Q_b 0
     0   0   1]
L1= [H_1 C 0_{2x1}]
L2= [sI_2 0 x^{pos}] 
R = [R_1 0   0
     0   R_2 0
     0   0   1]

'''

state_sz_src = 4
bias_sz_src = 2
observation_sz = 2
observations_1_biased = np.empty((observation_sz, 0))
observations_2_scaled = np.empty((observation_sz, 0))
estimated_state_array = np.empty((state_sz_src + bias_sz_src + 1, 0))
bias_ground_true = np.empty((bias_sz_src, 0))

"""
initial problem statement
"""
dt = 0.5 #time step of mobile movement
x = np.zeros((state_sz_src, 1))
b = np.zeros((bias_sz_src, 1))
s = np.ones((1, 1))
# b = 0.1*np.ones((2, 1))
P_x = 1.0 * np.eye(state_sz_src)
P_b = 1.2 * np.eye(bias_sz_src)

A = np.eye(state_sz_src)
A[0, 2] = dt
A[1, 3] = dt
# B = 0.125*np.eye(state_sz, bias_sz)
B = np.zeros((state_sz_src, bias_sz_src))
Q_x = 0.25 * np.eye(state_sz_src)
Q_b = 1.0 * np.eye(bias_sz_src)

y = np.zeros((observation_sz, 1))
H_1 = np.eye(observation_sz, state_sz_src)
H_2 = np.eye(observation_sz, state_sz_src)
C = np.eye(observation_sz, bias_sz_src)
# C = np.zeros((2, state_sz))
R_1 = 0.1*np.eye(observation_sz)
R_2 = 0.1*np.eye(observation_sz)

"""
partitioned Kalman estimator
"""
z = np.vstack((x, b, s))

F = np.eye((state_sz_src + bias_sz_src + 1))
F[:state_sz_src, :state_sz_src] = A

Q = np.eye((state_sz_src + bias_sz_src + 1))
Q[0:state_sz_src, 0:state_sz_src] = Q_x
Q[state_sz_src:state_sz_src + bias_sz_src, state_sz_src:state_sz_src + bias_sz_src] = Q_b

L_1 = np.hstack((H_1, C, np.zeros((2, 1))))
L_2 = np.zeros((observation_sz, state_sz_src + bias_sz_src + 1))
# L_2 = np.hstack((z[-1, 0]*np.eye(observation_sz), np.zeros((bias_sz_src, state_sz_src)), z[:observation_sz].reshape(2, 1)))
s = z[-1, 0]
L_2[0, 0] = s
L_2[1, 1] = s

P = np.zeros((state_sz_src + bias_sz_src + 1, state_sz_src + bias_sz_src + 1))
P[:state_sz_src, :state_sz_src] = P_x
P[state_sz_src:state_sz_src + bias_sz_src, state_sz_src:state_sz_src + bias_sz_src] = P_b
P[-1, -1] = 0.1
I = np.eye(state_sz_src + bias_sz_src + 1)

R = np.zeros((2*observation_sz, 2*observation_sz))
R[:observation_sz, :observation_sz] = R_1
R[observation_sz:2*observation_sz, observation_sz:2*observation_sz] = R_2

"""
generate trajectory, biases and set initial values
"""
N_iter = 500
[points_1, points_2] = generate_trajectories(num_samples=N_iter, trajectory_noise=(0.005, 0.005), noise_mult=(0.017, 0.019), scale=5.0)
bias_direction = np.array([[1.0], [0.5]])
ts = np.arange(0.0, N_iter)
bias_x = get_smoothed_piecewise_bias(jumps_number=(4, 10), mult=0.35, smooth_factor=(25, 15), savgol_params=(35, 3))
bias_y = get_smoothed_piecewise_bias(jumps_number=(4, 10), mult=0.35, smooth_factor=(25, 15), savgol_params=(35, 3))
bias = np.vstack((bias_x, bias_y))
points_1_biased = points_1 + bias
points_2_scaled = 0.6*points_2

z[0: 2] = 0.5*(get_closest_point(z[0: 2], points_1_biased) + get_closest_point(z[0: 2], points_2)).reshape(2, 1)
z[2] = -0.1
z[3] = 0.1


plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.plot(points_1[0], points_1[1])
plt.plot(points_2[0], points_2[1])
plt.subplot(2, 2, 2)
plt.plot(ts, bias_x, color='red')
plt.subplot(2, 2, 3)
plt.plot(points_1_biased[0], points_1_biased[1], color='red')
plt.plot(points_2_scaled[0], points_2_scaled[1], color='green')
plt.subplot(2, 2, 4)
plt.plot(ts, bias_y, color='red')
plt.tight_layout()
plt.show()


print('===========================================================')
for i in np.arange(0, N_iter):
    (z__, P) = kf_predict(z, P, F, Q, np.eye(state_sz_src), np.zeros((state_sz_src, 1)))

    P_inv = np.linalg.inv(P) + np.matmul(L_1.T, np.matmul(np.linalg.inv(R_1), L_1)) + np.matmul(L_2.T, np.matmul(np.linalg.inv(R_2), L_2))
    P = np.linalg.inv(P_inv)

    y_1, y_1_idx = get_closest_point_iterative(z[0:2], z[2:4], dt, points_1_biased)
    y_1 = y_1.reshape(2, 1)
    # y_2, _ = get_closest_point_iterative(z[0:2]/z[-1], z[2:4]/z[-1], dt, points_2_scaled)
    y_2, _ = get_closest_point_iterative(z[0:2], z[2:4], dt, points_2_scaled)
    y_2 = y_2.reshape(2, 1)

    estimated_state_array = np.hstack((estimated_state_array, z))
    observations_1_biased = np.hstack((observations_1_biased, y_1))
    observations_2_scaled = np.hstack((observations_2_scaled, y_2))
    bias_ground_true = np.hstack((bias_ground_true, bias[:, y_1_idx].reshape(2, 1)))

    plt.figure(figsize=(20, 10))
    plt.plot(points_1_biased[0], points_1_biased[1], color='red')
    plt.plot(points_2_scaled[0], points_2_scaled[1], color='green')
    plt.plot(points_2[0], points_2[1], color='grey')
    plt.plot(estimated_state_array[0], estimated_state_array[1], color='grey', linewidth=3)
    plt.scatter(observations_1_biased[0, -1], observations_1_biased[1, -1], c='r')
    plt.scatter(observations_2_scaled[0, -1], observations_2_scaled[1, -1], c='g')
    plt.arrow(z[0, 0], z[1, 0], 0.5*z[2, 0], 0.5*z[3, 0])
    plt.scatter(z__[0, 0], z__[1, 0], c='k', s=35)
    plt.scatter(z[0, 0], z[1, 0], c='grey', s=35)
    plt.scatter(y_1[0, 0], y_1[1, 0], c='r')
    plt.scatter(y_2[0, 0], y_2[1, 0], c='g')
    plt.show()

    L_2[0, 0] = z__[-1]
    L_2[1, 1] = z__[-1]
    # L_2[0, 6] = z__[0]
    # L_2[1, 6] = z__[1]

    K_1 = np.matmul(P, np.matmul(L_1.T, np.linalg.inv(R_1)))
    K_2 = np.matmul(P, np.matmul(L_2.T, np.linalg.inv(R_2)))

    z = z__ + np.matmul(K_1, (y_1 - np.matmul(L_1, z__))) + np.matmul(K_2, (y_2 - np.matmul(L_2, z__)))
    # plt.plot(estimated_state_array[0], estimated_state_array[1])
    # plt.show()
    print(z.flatten())
    print('===========================================================')