import numpy as np
import matplotlib.pyplot as plt


def gaussian_pdf(x, mu=0, sigma=1, multiplier=1):
    return multiplier / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def laplace_pdf(x, mu=0, b=1, multiplier=1):
    return multiplier / (2 * b) * np.exp(-np.abs(x - mu) / b)


def sample_pdf(arg_range, step, func):
    func_knots = np.empty((2, 0))
    for x in np.arange(arg_range[0], arg_range[1], step):
        knot = np.array([x, func(x)])
        func_knots = np.hstack((func_knots, knot.reshape(2, 1)))
    return func_knots


def func_sum(x):
    mu_1 = 0
    mu_2 = 5
    multiplier_1 = 0.005
    multiplier_2 = 0.995
    return laplace_pdf(x, mu=mu_1, b=1.0, multiplier=multiplier_1) + gaussian_pdf(x, mu=mu_2, sigma=0.025, multiplier=multiplier_2)


def func_laplace_product(x):
    mu_1 = 0
    mu_2 = 5
    multiplier_1 = 1.0
    multiplier_2 = 1.0
    return laplace_pdf(x, mu=mu_1, b=1.0, multiplier=multiplier_1) * laplace_pdf(x, mu=mu_2, b=0.85, multiplier=multiplier_2)


interval = [-3.5, 8.5]
sample_step = 0.01
pdf_knots = sample_pdf(interval, sample_step, func_sum)

from tools.class_distribution_function import *

'''
samples_number = int(1e3)
distribution = DistributionFunction(func=func_sum_2, interval=interval)
samples_list = distribution.generate_samples(samples_number)
'''

bins_number = 50
'''
fig, axs = plt.subplots(1, 2, tight_layout=True)
axs[0].plot(pdf_knots[0], pdf_knots[1])
axs[1].hist(samples_list, bins=bins_number)
plt.show()
'''
'''
x = np.linspace(0, samples_number - 1, samples_number)

plt.stem(x, samples_list, markerfmt=" ")
plt.tight_layout()
plt.show()
'''

x = np.linspace(-5, 30, int(1e4))
y = func_laplace_product(x)

plt.plot(x, y)
plt.show()