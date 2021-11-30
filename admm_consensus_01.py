import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt

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


length = 50
num_samples = 9*(length+1)

noise_amplitude_1 = 0.04
noise_amplitude_2 = 0.035
bias_mult = 0.6
biases_number_jumps = np.random.randint(5, 15)
biases_values = np.random.randn(biases_number_jumps)
biases_durations = np.abs(np.random.randn(biases_number_jumps))
biases_durations_normalized = biases_durations / np.sum(biases_durations)
noise_1 = noise_amplitude_1 * np.random.normal(0, 1, size=num_samples)
noise_2 = noise_amplitude_2 * np.random.normal(0, 1, size=num_samples)

xs = np.linspace(0, length, num_samples)
bias = get_biases(num_samples, biases_values, biases_durations_normalized)
sine = np.sin(0.5 * xs)
biased_noisy_sine = sine + bias + noise_1
noisy_sine = sine + noise_2
'''
plt.figure(figsize=(20, 10))
# plt.plot(x, sine, color='green')
plt.plot(xs, biased_noisy_sine, color='red')
plt.plot(xs, noisy_sine, color='green')
plt.plot(xs, bias, color='blue')
plt.tight_layout()
plt.show()
'''

lamda_1 = 0.5
lamda_2 = 1.5

x = cp.Variable(num_samples)
funcs = [cp.sum_squares(x - noisy_sine), lamda_1*cp.sum_squares(x - biased_noisy_sine), lamda_2*cp.norm(x, 2)]
prob = cp.Problem(cp.Minimize(sum(funcs)))
result = prob.solve()


plt.figure(figsize=(20, 10))
# plt.plot(x, sine, color='green')
plt.plot(xs, biased_noisy_sine, color='red')
plt.plot(xs, noisy_sine, color='green')
plt.plot(xs, x.value, linewidth=6, color='blue')
plt.tight_layout()
plt.show()