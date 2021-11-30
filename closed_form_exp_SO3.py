import numpy as np
from scipy.spatial.transform import Rotation
import math

def alignVectors(vec1, vec2):
    n1 = vec1 / np.linalg.norm(vec1)
    n2 = vec2 / np.linalg.norm(vec2)
    v = np.cross(n1, n2)
    vhat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    s = np.linalg.norm(v)
    c = np.dot(n1, n2)
    R = np.identity(3) + vhat + np.matmul(vhat, vhat) * (1 - c) / s / s
    return R


def alignVectors2(vec1, vec2):
    n1 = vec1 / np.linalg.norm(vec1)
    n2 = vec2 / np.linalg.norm(vec2)
    v = np.cross(n1, n2)
    vhat = np.array([[0,     v[2], -v[1]],
                     [-v[2], 0,     v[0]],
                     [v[1], -v[0],    0]])

    vhat *= -1
    sin_theta = np.linalg.norm(v)
    cos_theta = np.dot(n1, n2)
    theta = math.acos(cos_theta)
    theta__ = math.asin(sin_theta)
    S = vhat #/ (sin_theta)

    R = np.identity(3) + S + (1-cos_theta) * np.matmul(S, S) / (sin_theta*sin_theta)
    return R

v1 = np.array([1, 0, 0])

for idx in range(100):
    random_angles = np.random.rand(3, 1)
    rotation_random = Rotation.from_euler('xyz', random_angles.reshape(3, ))
    R_random = rotation_random.as_matrix()
    v2 = np.matmul(R_random, v1)
    R_check = alignVectors2(v1, v2)
    # print(R_random - R_check)
    v2_check = np.matmul(R_check, v1)
    print(v2_check - v2)
