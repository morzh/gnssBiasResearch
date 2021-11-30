import matplotlib.pyplot as plt
import numpy as np
import os

path = '/home/morzh/work/DSOPP_tests_data_temp/proto_track/bias_iterations'
iterations = 500
for idx in range(iterations):
    if idx % 10 != 0:
        continue
    filename = 'bias.' + str(idx) + '.npy'
    filepath = os.path.join(path, filename)
    bias = np.load(filepath)
    bias_flatten = bias.flatten()
    bias = bias_flatten.reshape(3, -1, order="F")
    bias_norms = np.linalg.norm(bias, axis=0)

    if idx > 0:
        bias_norms_abs_diff = np.abs(bias_norms - bias_norms_prev)
        print('iteration number', idx)
        print(np.sum(bias_norms_abs_diff))

    xs = np.linspace(0, bias_norms.size-1, bias_norms.size)

    plt.figure(figsize=(20, 10))
    plt.plot(xs, bias_norms)
    plt.tight_layout()
    plt.show()

    bias_norms_prev = bias_norms