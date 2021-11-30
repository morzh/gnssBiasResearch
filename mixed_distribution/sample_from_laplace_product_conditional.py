import numpy as np
from tools.class_distribution_function import *
import matplotlib.pyplot as plt

def product_laplace_minus_inf_plus_inf(x, mu=1.1, b1=1.0, b2=0.9):
    normalization_divisor = b1*b2/(b1+b2)*(np.exp(-mu/b1) + np.exp(-mu/b2)) - b1*b2/(b2-b1)*(np.exp(-mu/b1) - np.exp(-mu/b2))
    return np.exp(-np.abs(x)/b1 - np.abs(x-mu)/b2) / normalization_divisor

samples_number = 300
interval = [-2.5, 2.5]
samples = []

'''
for idx in range(samples_number):
    distribution = DistributionFunction(func=product_laplace_minus_inf_plus_inf, interval=interval)
    sample = distribution.generate_samples(1)
    samples.append(sample)
'''


distribution = DistributionFunction(func=product_laplace_minus_inf_plus_inf, interval=interval)
samples_list = distribution.generate_samples(samples_number)

x = np.linspace(0, samples_number - 1, samples_number)

plt.stem(x, samples_list, markerfmt=" ")
plt.tight_layout()
plt.show()
