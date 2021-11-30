'''
Bernard Friedland
Treatment of Bias in Recursive Filtering
'''
from KF.KF_utils import *

dt = 0.1# time step of mobile movement

'''
Initialization of state matrices
'''
x = np.array([[0.0], [0.0], [0.1], [0.1]]) #(nx1) mean state estimate of the previous step (k−1)
u = np.zeros((x.shape[0], 1)) #control input
b = np.zeros((2, 1)) #(rx1) bias

A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]) #transition (nxn) matrix.
B = np.eye(x.shape[0], 2) #bias effect matrix
U = np.eye(x.shape[0]) #input effect matrix

P = np.diag((0.01, 0.01, 0.01, 0.01)) #state covariance of previous step (k−1)
Q = np.eye(x.shape[0]) #process noise covariance matrix

'''
Measurement matrices
'''
y = np.array([[x[0, 0] + abs(np.random.randn(1)[0])], [x[1, 0] + abs(np.random.randn(1)[0])]]) #measurement vector

H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) #measurement matrix
C = np.eye(2, 2) #bias effect matrix

R = np.eye(y.shape[0]) #measurement covariance matrix


'''
New state matrices
'''
z = np.vstack((x, b))
u1 = np.vstack((u, np.zeros(b.shape)))

F = np.vstack((np.hstack((A, B)), np.zeros((b.shape[0], A.shape[1]+B.shape[1]))))
# F = np.vstack((np.hstack((A, U)), np.hstack((np.zeros(A.shape), np.eye(U.shape[0])))))
G = np.vstack((np.eye(x.shape[0]), np.zeros((b.shape[0], x.shape[0]))))
L = np.hstack((H, C))
U1 = np.zeros((z.shape[0], z.shape[0]))
U1[0:U.shape[0], 0:U.shape[1]] = U

P_tilda = np.vstack((np.hstack((P, np.zeros((P.shape[0], b.shape[0])))), np.hstack((np.zeros((b.shape[0], P.shape[0])), 0.1*np.eye(b.shape[0])))))

#Number of iterations in Kalman Filter
N_iter = 50

#Applying the Kalman Filter
for i in np.arange(0, N_iter):
    (z, P_tilda) = kf_predict(z, P_tilda, F, np.dot(G, Q), U1, u1)
    print(z)
    print(P_tilda)
