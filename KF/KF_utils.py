import numpy as np
from numpy.linalg import inv
from scipy import interpolate


def get_biases(num_samples, biases_values, biases_durations_normalized):
    biases_timeline = np.array([0])
    biases_durations = num_samples*biases_durations_normalized
    x = np.empty((0, 1))
    for idx in range(biases_durations.shape[0]):
        biases_timeline = np.vstack((biases_timeline, biases_timeline[-1]+biases_durations[idx]))

    for idx in range(biases_timeline.shape[0]-1):
        x = np.vstack((x, biases_values[idx]*np.ones((round(float(biases_timeline[idx+1])) - round(float(biases_timeline[idx])), 1))))

    if x.shape[0] < num_samples:
        x = np.vstack((x, x[-1]*np.ones(num_samples-x.shape[0])))
    return x.reshape(num_samples,)

def get_trajectory(num_samples = 500, noise_mult=3e-3):
    arr = np.array([[0, 0], [2, 0.5], [2.5, 1.25], [2.6, 2.8], [1.3, 1.1]])
    x, y = zip(*arr)
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]
    f, u = interpolate.splprep([x, y], s=0, per=True)
    xint, yint = interpolate.splev(np.linspace(0, 1, num_samples), f)
    xint += noise_mult*np.random.randn(num_samples)
    yint += noise_mult*np.random.randn(num_samples)
    return np.vstack((xint, yint))

def gauss_pdf(X, M, S):
    if M.shape[1] == 1:
        DX = X - np.tile(M, X.shape[1])
        E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
        P = np.exp(-E)
    elif X.shape[1] == 1:
        DX = np.tile(X, M.shape[1]) - M
        E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
        P = np.exp(-E)
    else:
        DX = X - M
        E = 0.5 * np.dot(DX.t, np.dot(inv(S), DX))
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
        P = np.exp(-E)
    return (P[0], E[0])


def kf_predict(x, P, A, Q, B, U):
    x = np.dot(A, x)# + np.dot(B, U)
    P = np.dot(A, np.dot(P, A.T)) + Q
    return (x, P)

'''
def kf_bias_predict(X, P, A, Q, B, Bb, U):
    X = np.dot(A, X) + np.dot(Bb, X)  + np.dot(B, U)
    P =
    return (X, P)
'''

def kf_update(X, P, Y, H, R):
    IM = np.dot(H, X)
    IS = R + np.dot(H, np.dot(P, H.T))
    K = np.dot(P, np.dot(H.T, inv(IS)))
    X = X + np.dot(K, (Y-IM))
    P = P - np.dot(K, np.dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return (X, P, K, IM, IS, LH)


def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)
