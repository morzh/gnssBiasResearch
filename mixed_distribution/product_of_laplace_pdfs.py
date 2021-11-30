import numpy as np
import matplotlib.pyplot as plt


def gaussian_pdf(x, mu=0, sigma=1, multiplier=1):
    return multiplier / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def laplace_pdf(x, mu=0, b=1, multiplier=1):
    return multiplier / (2 * b) * np.exp(-np.abs(x - mu) / b)

def func_laplace_product(x):
    mu_1 = 0
    mu_2 = 10
    b_1 = 1.0
    b_2 = 1.0
    multiplier_1 = 1.0
    multiplier_2 = 1.0
    return laplace_pdf(x, mu=mu_1, b=b_1, multiplier=multiplier_1) * laplace_pdf(x, mu=mu_2, b=b_2, multiplier=multiplier_2)


b_prev = 5
b_1 = 0.7#diversity
b_2 = 0.85#diversity


x = np.linspace(-5, 20, int(1e4))
y = laplace_pdf(x, b=b_1) * laplace_pdf(x, mu=b_prev, b=b_2)

fig, ax = plt.subplots()
textstr = '\n'.join((r'$b_{i-1}=%.2f$' % (b_prev, ), r'$c_1=%.2f$' % (b_1, ), r'$c_2=%.2f$' % (b_2, )))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,  verticalalignment='top', bbox=props)
ax.plot(x, y)
plt.show()