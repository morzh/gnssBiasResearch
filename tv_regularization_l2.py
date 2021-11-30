# sphinx_gallery_thumbnail_number = 5
import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')
np.random.seed(1)

nx = 101
x = np.zeros(nx)
x[:nx//2] = 10
x[nx//2:3*nx//4] = -5

x = np.linspace(0, 100, nx)
x = np.sin(0.5 * x)

Iop = pylops.Identity(nx)
n = 0.13*np.random.normal(0, 1, nx)
y = Iop*(x + n)

plt.figure(figsize=(10, 5))
plt.plot(x, 'k', lw=3, label='x')
plt.plot(y, '.k', label='y=x+n')
plt.legend()
plt.title('Model and data')
plt.show()


D2op = pylops.SecondDerivative(nx, edge=True)
lamda = 1e2
xinv = pylops.optimization.leastsquares.RegularizedInversion(Iop, [D2op], y, epsRs=[np.sqrt(lamda/2)], **dict(iter_lim=30))

plt.figure(figsize=(10, 5))
plt.plot(x, 'k', lw=3, label='x')
plt.plot(y, '.k', label='y=x+n')
plt.plot(xinv, 'r', lw=5, label='xinv')
plt.legend()
plt.title('L2 inversion')
plt.show()