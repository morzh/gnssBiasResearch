
import numpy as np
from matplotlib import pyplot as plt
import pylops

from skimage.restoration import denoise_tv_chambolle

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


num_samples = 9*51
noise_amplitude_1 = 0.04
noise_amplitude_2 = 0.035
bias_mult = 0.6
biases_number_jumps = np.random.randint(5, 15)
biases_values = np.random.randn(biases_number_jumps)
biases_durations = np.abs(np.random.randn(biases_number_jumps))
biases_durations_normalized = biases_durations / np.sum(biases_durations)
noise_1 = noise_amplitude_1 * np.random.normal(0, 1, size=num_samples)
noise_2 = noise_amplitude_2 * np.random.normal(0, 1, size=num_samples)


length = 50
x = np.linspace(0, length, num_samples)
bias = get_biases(num_samples, biases_values, biases_durations_normalized)
sine = np.sin(0.5*x)
# noisy_sine = bias_mult*bias
# noisy_sine = sine + bias_mult*bias + noise

# bias = denoise_tv_chambolle(bias_mult*bias, weight=1.2)
biased_noisy_sine = sine + bias + noise_1
noisy_sine = sine + noise_2
sine_residuals = biased_noisy_sine - noisy_sine


Iop = pylops.Identity(num_samples)
y = Iop*(sine_residuals)

Dop = pylops.FirstDerivative(num_samples, edge=True, kind='backward')
mu = 0.07
lamda = 0.1
niter_out = 150
niter_in = 3

xinv, niter = pylops.optimization.sparsity.SplitBregman(Iop, [Dop], y, niter_out,
                                              niter_in, mu=mu, epsRL1s=[lamda],
                                              tol=1e-4, tau=1.,
                                              **dict(iter_lim=30, damp=1e-10))


plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
# plt.plot(x, sine, color='green')
plt.plot(x, biased_noisy_sine, color='red')
plt.plot(x, noisy_sine, color='green')
# plt.plot(x, bias, color='blue')
# plt.plot(x, sine_residuals, color='black')
plt.tight_layout()
# plt.show()
plt.subplot(2, 1, 2)
# plt.figure(figsize=(10, 5))
# plt.plot(y, '.k', label='y=x+n')
plt.plot(x, xinv, 'r', lw=5, label='Estimated bias')
plt.plot(x, bias, color='blue', label='GT bias')
plt.legend()
# plt.title('TV inversion')
plt.show()