import numpy as np
import matplotlib.pyplot as plt
import skfda
from skfda.preprocessing.registration.elastic import elastic_mean
from skfda.preprocessing.registration import ElasticRegistration

"""
A. Srivastava†, W. Wu, S. Kurtek, E. Klassen, and J. S. Marron
Registration of Functional Data Using Fisher-Rao Metric
page 
"""


def piecewise_linear(xs, xs_src, ys_src):
    ys = np.zeros(xs.shape)
    for idx in range(xs.shape[0]-1):
        idx_second = np.argmin(xs[idx] >= xs_src)
        idx_first = np.clip(idx_second-1, 0, xs.shape[0]-2)
        lamda = (xs[idx] - xs_src[idx_first]) / (xs_src[idx_second] - xs_src[idx_first])
        ys[idx] = (1 - lamda) * ys_src[idx_first] + lamda * ys_src[idx_second]

    ys[-1] = ys_src[-1]
    return ys

num_functions = 21
num_points = 50
a = np.linspace(-1, 1, num_functions)
funcs = list()
t = np.linspace(-3, 3, num_points)

plt.figure(figsize=(20, 10))
for idx in range(num_functions):
    func = np.zeros((2, num_points))
    z_1 = np.random.normal(1.0, 0.25)
    z_2 = np.random.normal(1.0, 0.25)

    if np.abs(a[idx]) < 1e-6:
        gamma = t
    else:
        gamma = 6 * ((np.exp(a[idx] * (t + 3) / 6) - 1) / (np.exp(a[idx])-1)) - 3

    y = z_1 * np.exp(-(t - 1.5)**2/2) + z_2 * np.exp(-(t + 1.5)**2/2)
    func[0] = gamma
    func[1] = y
    funcs.append(func)

    plt.subplot(2, 2, 1)
    plt.plot(t, y)
    plt.subplot(2, 2, 2)
    plt.plot(t, gamma)
    plt.subplot(2, 2, 3)
    plt.plot(gamma, y)

plt.tight_layout()
# plt.show()

grid_points = list()
data_matrix = list()

plt.subplot(2, 2, 4)
for fn in funcs:
    y = piecewise_linear(t, fn[0], fn[1])
    data_matrix.append(y.tolist())
    plt.plot(t, y)
plt.show()


data_matrix = np.expand_dims(data_matrix, axis=2)
fd = skfda.representation.FDataGrid(grid_points=t, data_matrix=data_matrix)
karcher_mean = elastic_mean(fd)

karcher_mean.plot(linewidth=10)
for fn in funcs:
    plt.plot(fn[0], fn[1])
plt.show()


elastic_registration = ElasticRegistration()
fd_align = elastic_registration.fit_transform(fd)
fd_align.plot()
plt.show()
