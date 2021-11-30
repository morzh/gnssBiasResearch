import matplotlib.pyplot as plt
from KF.KF_utils import *


observered_array = np.empty((2, 0))
estimated_state_array = np.empty((4, 0))

#time step of mobile movement
dt = 0.1

# Initialization of state matrices
X = np.array([[0.0], [0.0], [0.1], [0.1]])
P = np.diag((0.01, 0.01, 0.01, 0.01))
A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
Q = np.eye(X.shape[0])
B = np.eye(X.shape[0])
U = np.zeros((X.shape[0], 1))
# U[2:] = 0.1

# Measurement matrices
Y = np.array([[X[0, 0] + abs(np.random.randn(1)[0])], [X[1, 0] + abs(np.random.randn(1)[0])]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
R = np.eye(Y.shape[0])

# Number of iterations in Kalman Filter
N_iter = 50

# Applying the Kalman Filter
for i in np.arange(0, N_iter):
    (X, P) = kf_predict(X, P, A, Q, B, U)
    (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
    Y = np.array([[X[0, 0] + abs(0.1 * np.random.randn(1)[0])], [X[1, 0] + np.abs(0.1 * np.random.randn(1)[0])]])
    estimated_state_array = np.hstack((estimated_state_array, X))
    observered_array = np.hstack((observered_array, Y))

plt.plot(observered_array[0], observered_array[1], color='green', linewidth=4)
plt.plot(estimated_state_array[0], estimated_state_array[1], color='red')
plt.show()