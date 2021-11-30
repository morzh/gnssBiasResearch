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
n = 0.1*np.random.normal(0, 1, nx)
y = Iop*(x + n)

plt.figure(figsize=(10, 5))
plt.plot(x, 'k', lw=3, label='x')
plt.plot(y, '.k', label='y=x+n')
plt.legend()
plt.title('Model and data')
plt.show()


Dop = pylops.FirstDerivative(nx, edge=True, kind='backward')
mu = 0.01
lamda = 0.3
niter_out = 50
niter_in = 3

xinv, niter = \
    pylops.optimization.sparsity.SplitBregman(Iop, [Dop], y, niter_out,
                                              niter_in, mu=mu, epsRL1s=[lamda],
                                              tol=1e-4, tau=1.,
                                              **dict(iter_lim=30, damp=1e-10))

plt.figure(figsize=(10, 5))
plt.plot(x, 'k', lw=3, label='x')
plt.plot(y, '.k', label='y=x+n')
plt.plot(xinv, 'r', lw=5, label='xinv')
plt.legend()
plt.title('TV inversion')
plt.show()