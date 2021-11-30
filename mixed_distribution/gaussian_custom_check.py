import numpy as np
from numpy.random import default_rng
import scipy.stats as st

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


def gaussian_pdf(self, x, mu, sigma, multiplier=1):
    return multiplier / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


class gaussian_gen(st.rv_continuous):
    def _pdf(self, xs):
        mu = 0.0
        sigma = 1.0
        multiplier = 1.0
        return multiplier / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5*((xs - mu) / sigma) ** 2)

N = int(1e4)
bins_number = 80

gaussian = gaussian_gen(a=0, b=1, name='gaussian')
rng = default_rng()

gaussian_samples_numpy = np.random.normal(size=N)
# gaussian_samples_my = gaussian.rvs(size=N, random_state=rng)
gaussian_samples_my = gaussian.rvs(size=N)

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(gaussian_samples_numpy, bins=bins_number)
axs[1].hist(gaussian_samples_my, bins=bins_number)
plt.show()