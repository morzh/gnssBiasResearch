import numpy as np

kEigenValueThreshold = 1e-9

sigma = np.array([[4, 16, 0], [16, 64, 0], [0, 0, 9]])
# print(np.linalg.det(sigma))

sigma11 = np.array([[4, 0], [0, 9]])
B = np.array([[4.01], [0.01]])
# print(B)
# print(sigma11)

sigma11_B = np.dot(sigma11, B)
B = np.array([[3.99], [-0.01]])
Bt_sigma11t_B = np.dot(sigma11_B.T, B)

sigma__ = np.vstack((np.hstack((sigma11, sigma11_B)), np.hstack((sigma11_B.T, Bt_sigma11t_B))))

# print(sigma__)
# print(np.linalg.det(sigma__))

# print('---------------------------------------------------------')
A = np.array([[0.52194049338881243, 1.7100749243848732, 0.043584391176971791],
              [1.7100749243848732, 7.8330410372207595, -0.13696139822232137],
              [0.043584391176971791, -0.13696139822232137, 0.038733350696786635]])

# print(np.linalg.det(A))
w, v = np.linalg.eig(A)
# print(w)


A1 = A[:, :2].reshape(3, 2)
A2 = A[:, 2].reshape(3, 1)
B = np.linalg.pinv(A1) @ A2
# print(B)
# print(A1 @ B)
# print(A2)

stddev = 0.1

A11 = A[:2, :2].reshape(2, 2)
delta = 0.1 + np.random.normal(0.0, stddev, size=(2, 1))
B_augmented = B + delta
A11_B = A11 @ B_augmented
delta = np.random.normal(0.0, stddev, size=(2, 1))
# B_augmented = B + delta
Bt_A11t = B_augmented.T @ A11.T
delta = np.random.normal(0.0, stddev, size=(2, 1))
B_augmented = B + delta
Bt_A11t_B = Bt_A11t @ A11.T @ B_augmented
# Bt_A11t_B = B_augmented.T @ A11.T @ B_augmented


A__ = np.vstack((np.hstack((A11, A11_B)), np.hstack((A11_B.T, Bt_A11t_B))))

print(np.linalg.det(A__))
w, v = np.linalg.eig(A__)
print(w)
