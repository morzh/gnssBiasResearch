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
y_shift = 1

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
biased_noisy_sine = sine + bias + noise_1 + y_shift
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

lamda_1 = np.random.uniform(1, 2)
lamda_2 = np.random.uniform(1, 2)
lamda_3 = np.random.uniform(0.1, 5)

x = cp.Variable(num_samples)
b = cp.Variable(num_samples)
# funcs = [cp.sum_squares(x - noisy_sine), lamda_1*cp.sum_squares(x + b - biased_noisy_sine), lamda_3*cp.tv(b)]
funcs = [cp.sum_squares(x - noisy_sine), lamda_1*cp.sum_squares(x + b - biased_noisy_sine), lamda_2*cp.norm(x, 2), lamda_3*cp.tv(b)]
prob = cp.Problem(cp.Minimize(sum(funcs)))
result = prob.solve(verbose=True)

plt.figure(figsize=(20, 10))
textbox_objective_function = '$||sin(x)||_2^2 + \lambda_1  || \left [sin(x)+b(x) \\right ] ||_2^2 + \lambda_2 ||x||_2^2 + \lambda_3 || \\nabla b ||_1$'
textbox_lambda_values = '\n'.join((r'$\lambda_1=%.2f$' % (lamda_1,), r'$\lambda_2=%.2f$' % (lamda_2,), r'$\lambda_3=%.2f$' % (lamda_3,)))
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
plt.text(1.0, 1.5, textbox_objective_function, fontsize=15, verticalalignment='top', bbox=props)
plt.text(-1.0, -1.0, textbox_lambda_values, fontsize=15, verticalalignment='bottom', bbox=props)
plt.plot(xs, noisy_sine, linewidth=5, color='darkgray')
plt.plot(xs, biased_noisy_sine, linewidth=2, color='lightgray')
plt.plot(xs, bias, linewidth=5, color='darkgray')
plt.plot(xs, x.value, linewidth=2, color='green')
plt.plot(xs, b.value, linewidth=2, color='blue')
plt.legend(('noisy sine', 'noisy sine with bias', 'true bias', 'estimated sine', 'estimated bias'), loc='upper center', shadow=True)
plt.tight_layout()
plt.show()