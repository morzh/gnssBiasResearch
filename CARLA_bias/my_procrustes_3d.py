import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def my_procrustes_3d(points_to_align, points):
    mu_X = np.mean(points_to_align, axis=1).reshape(3, 1)
    mu_Y = np.mean(points, axis=1).reshape(3, 1)

    X_centered = points_to_align - mu_X
    Y_centered = points - mu_Y
    translate = mu_X-mu_Y
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X_centered_scaled[0], X_centered_scaled[1], X_centered_scaled[2])
    ax.plot(Y_centered[0], Y_centered[1], Y_centered[2])
    plt.show()
    '''
    k = points_to_align.shape[1]
    X0_sqrtsum = np.sum(np.linalg.norm(X_centered, axis=0)**2)
    Y0_sqrtsum = np.sum(np.linalg.norm(Y_centered, axis=0)**2)

    s1 = np.sqrt(X0_sqrtsum / k)
    s2 = np.sqrt(Y0_sqrtsum / k)
    scale = s1/s2

    X_centered_scaled = X_centered / s1
    Y_centered_scaled = Y_centered / s2
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X_centered_scaled[0], X_centered_scaled[1], X_centered_scaled[2])
    ax.plot(Y_centered[0], Y_centered[1], Y_centered[2])
    plt.show()
    '''
    U, s, Vt = np.linalg.svd(Y_centered_scaled @ X_centered_scaled.T)
    R = U @ Vt
    R_normalizer = np.eye(3)
    R_normalizer[2, 2] = np.linalg.det(U @ Vt)
    R_SO3 = U @ R_normalizer @ Vt

    X_rotated = R @ X_centered_scaled

    return X_rotated, Y_centered_scaled, s2/s1, R_SO3,  - (s2/s1)*R @ mu_X + mu_Y, R_normalizer
