import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from KF.KF_utils import *
from KF.generate_trajectory import *

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

"""
code written using paper:
"Fusion of IMU and Vision for Absolute Scale Estimation in Monocular SLAM"
by Gabriel Nutzi, Stephan Weiss, Davide Scaramuzza and Roland Siegwart
"""
"""
dynamics is:
z_{k+1} = F_k z_k + v_k 
z_k = [x_k.T, v_k.T, a_k.T, l_k].T

F_k = [ I_n    (T/l) I_n    (T^2/2l) I_n   -T/(l^2) v - T^2/(2 l^2) a 
          0          I_n           T I_n                             0
          0            0             I_n                             0
          0            0               0                             1 ]

y_{v,k}  = [I_3 0_3 0_3 0] z_k 
y_{I,k}  = [0_3 0_3 I_3 0] z_k 
"""

N = 2
dt = 0.2
state_sz = 3 * N + 1
observation_sz = 2

observations_1_vision = np.empty((observation_sz, 0))
observations_2_accel = np.empty((observation_sz, 0))
estimated_state_array = np.empty((state_sz, 0))

z = np.zeros((state_sz, 1))
# z[2] = -0.1
# z[3] = 0.1
# z[4] = -0.25
# z[5] = 0.25
z[6] = 1.2

F_base = np.eye(state_sz)
F_base[:N, N: 2 * N] = dt * np.eye(N)
F_base[:N, 2 * N: 3 * N] = 0.5 * dt ** 2 * np.eye(N)
F_base[N: 2 * N, 2 * N: 3 * N] = dt * np.eye(N)

P = 0.2*np.eye(state_sz)
Q = np.eye(state_sz)

y_v = np.zeros((observation_sz, 1))
y_I = np.zeros((observation_sz, 1))

H_v = np.eye(N, state_sz)
# H_1[0:N, 0:N] = np.eye(N)

H_i = np.zeros((N, state_sz))
H_i[0:N, 2 * N: 3 * N] = np.eye(N)

R_v = 0.2*np.eye(N)
R_i = 0.2*np.eye(N)

N_iter = 500
[points_1, points_2, noise_1, noise_2] = generate_trajectories(num_samples=N_iter, trajectory_noise=(0.005, 0.005), noise_mult=(0.017, 0.019), scale=5.0, separate_noise=True)
points_2_scaled = 0.6*points_2
points_2_noise_scaled = 0.6*points_2_scaled + noise_2

points_noise_1 = points_1 + noise_1
points_noise_2 = points_2 + noise_2

closest_point, idx = get_closest_point(z[:N], points_noise_1)
z[:N] = closest_point.reshape(2, 1)
z[N: 2*N] = ((points_noise_1[:, N_iter % idx+1] - points_noise_1[:, idx])/dt).reshape(2, 1)
speed = 2 #points per step

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(points_1[0], points_1[1])
plt.plot(points_2[0], points_2[1])
plt.plot(points_noise_1[0], points_noise_1[1])
plt.plot(points_noise_2[0], points_noise_2[1])
plt.subplot(1, 2, 2)
plt.plot(points_noise_1[0], points_noise_1[1])
plt.plot(points_2_noise_scaled[0], points_2_noise_scaled[1])
plt.show()

print('===========================================================')

for i in range(0, N_iter, speed):

    F = np.copy(F_base)
    F[:N, N: 2 * N] /= z[-1]
    F[:N, 2 * N: 3 * N] /= z[-1]
    F[:N, -1] = -dt / z[-1] ** 2 * (z[N: 2 * N] + 0.5 * z[2 * N: 3 * N]).reshape(2, )

    (z__, P) = kf_predict(z, P, F, Q, np.eye(state_sz), np.zeros((state_sz, 1)))

    pos_slam = points_1[:, i]
    pos_slam = pos_slam.reshape(2, 1)
    a_imu = get_acceleration(points_2_scaled, i, dt)
    a_imu = a_imu.reshape(2, 1)

    estimated_state_array = np.hstack((estimated_state_array, z))
    observations_1_vision = np.hstack((observations_1_vision, pos_slam))
    observations_2_accel = np.hstack((observations_2_accel, a_imu))

    P_mult_H_i = np.matmul(P, H_i.T)
    P_mult_H_v = np.matmul(P, H_v.T)
    K_v = np.matmul(P_mult_H_v,  np.linalg.inv(np.matmul(H_v, P_mult_H_v) + R_v))
    K_i = np.matmul(P_mult_H_i,  np.linalg.inv(np.matmul(H_i, P_mult_H_i) + R_i))

    z = z__ + np.matmul(K_v, pos_slam - np.matmul(H_v, z__))
    z = z__ + np.matmul(K_i, a_imu - np.matmul(H_i, z__))

    P = np.matmul(np.eye(state_sz) - np.matmul(K_v, H_v), P)
    P = np.matmul(np.eye(state_sz) - np.matmul(K_v, H_v), P)
    '''
    plt.figure(figsize=(20, 10))
    plt.plot(points_1[0], points_1[1], color='red')
    plt.plot(points_2_scaled[0], points_2_scaled[1], color='green')
    plt.plot(estimated_state_array[0], estimated_state_array[1], color='grey', linewidth=3)
    plt.scatter(observations_1_vision[0, -1], observations_1_vision[1, -1], c='r')
    plt.scatter(observations_2_accel[0, -1], observations_2_accel[1, -1], c='g')
    plt.arrow(z[0, 0], z[1, 0], 0.5*z[2, 0], 0.5*z[3, 0])
    plt.scatter(z__[0, 0], z__[1, 0], c='k', s=35)
    plt.scatter(z[0, 0], z[1, 0], c='grey', s=35)
    plt.scatter(pos_slam[0, 0], pos_slam[1, 0], c='r')
    # plt.scatter(y_2[0, 0], y_2[1, 0], c='g')
    plt.show()
    '''

    print(z.flatten())
    print('===========================================================')



