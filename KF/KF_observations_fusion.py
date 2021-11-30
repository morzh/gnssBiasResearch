import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from KF.KF_utils import *

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


"""
Shu-li Sun
Multi-sensor optimal information fusion Kalman filters with applications
dynamics is:
x(t+1) = Φ x(t) + Γ w(t)
y_i(t)  = H_i x(t) + v_i(t), i = 1, 2, ..., l
"""

state_sz = 4
true_position_array = np.empty((2, 0))
estimated_state_array = np.empty((2*state_sz, 0))
state_covariance_bias_array = np.empty((2, 0))


dt = 0.1# time step of mobile movement
x = np.zeros((state_sz, 1))
x[2:state_sz] = 0.1
P = 1.0 * np.eye(state_sz)

A = np.eye(state_sz)
Q = 0.25 * np.eye(state_sz)

y = np.zeros((2, 1))
H = np.eye(2, state_sz)
C = np.eye(2, state_sz)
# C = np.zeros((2, state_sz))
R = 0.1*np.eye(2)


print('===========================================================')

for i in np.arange(0, N_iter):
    (z, P_tilde) = kf_predict(z, P_tilde, F, Q, np.eye(2*state_sz), np.zeros((2*state_sz, 1)))
    (z, P_tilde, _, _, _, _) = kf_update(z, P_tilde, y, L, R)

    estimated_state_array = np.hstack((estimated_state_array, z))
    true_position_array = np.hstack((true_position_array, y))
    state_covariance_bias_array = np.hstack((state_covariance_bias_array, np.array([[P_tilde[4, 4]], [P_tilde[5, 5]]])))
    noise = noise_mult*abs(np.random.randn(2))
    y = z[0:2].reshape(2, 1) + noise.reshape(2, 1) + biases[i]*np.array([[1], [1]])
    # y = z[0:2] + noise.reshape(2, 1) + bias.reshape(2, 1)
    # plt.plot(estimated_state_array[0], estimated_state_array[1])
    # plt.show()
    # print(noise)
    print(z.flatten())
    # print(P_tilde)
    print('===========================================================')
