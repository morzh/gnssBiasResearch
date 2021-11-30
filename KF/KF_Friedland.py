import numpy as np
import matplotlib.pyplot as plt


"""
dynamics is:
x_k = A_{k-1} x_{k-1}  + B_{k-1} b_{k-1} + \\xi_{k-1}
b_{k+1} = b_k

y_k = H_k x_k + C_k b_k + \eta_{k}
"""

dt = 0.1 # time step of mobile movement

x = np.zeros((2, 1))
b = np.zeros((2, 1))
# b = 0.1*np.ones((2, 1))
P = 0.2*np.eye(2)

A = np.eye(2)
B = np.eye(2)
Q = 1.5*np.eye(2)

y = np.zeros((2, 1))
H = np.eye(2)
# C = np.eye(2)
C = np.zeros((2, 2))
R = 1.5*np.eye(2)


z = np.vstack((x, b))
# F = np.vstack((np.hstack((A, B)), np.hstack((np.zeros(A.shape), np.zeros((B.shape[0], B.shape[0]))))))
F = np.vstack((np.hstack((A, B)), np.hstack((np.zeros(A.shape), np.eye(B.shape[0])))))
G = np.vstack((np.eye(x.shape[0]), np.zeros((b.shape[0], x.shape[0]))))
L = np.hstack((H, C))
P_tilde = np.vstack((np.hstack((P, np.zeros((P.shape[0], b.shape[0])))), np.hstack((np.zeros((b.shape[0], P.shape[0])), 0.1*np.eye(b.shape[0])))))
I = np.eye(4)

# Simulation
y = np.array([[z[0, 0] + abs(np.random.randn(1)[0])], [z[1, 0] + abs(np.random.randn(1)[0])]]) + np.array([[0.1], [0.1]])
N_iter = 150
print(z.T)
print(P_tilde)
print('===========================================================')
for i in np.arange(0, N_iter):
    #state predict and update
    S = np.matmul(L, np.matmul(P_tilde, L.T)) + R
    K = np.matmul(np.matmul(P_tilde, L.T), np.linalg.inv(S))
    t = y - np.dot(np.matmul(L, F), z)
    z = np.dot(F, z) + np.matmul(K, t)
    #state covariance predict and update
    T = np.matmul(I - np.matmul(K, L), P_tilde)
    P_tilde = np.matmul(F, np.matmul(T, F.T)) + np.matmul(G, np.matmul(Q, G.T))

    y = np.array([[z[0, 0] + abs(0.1 * np.random.randn(1)[0])], [z[1, 0] + np.abs(0.1 * np.random.randn(1)[0])]]) + np.array([[0.1], [0.1]])
    print(z.T)
    # print(P_tilde)
    print('===========================================================')