# sphinx_gallery_thumbnail_number = 2
import numpy as np
import matplotlib.pyplot as plt
import pylops

from scipy.sparse.linalg import lsqr


plt.close('all')
np.random.seed(10)

def prior_realization(f0, a0, phi0, sigmaf, sigmaa, sigmaphi, dt, nt, nfft):
    """Create realization from prior mean and std for amplitude, frequency and
    phase
    """
    f = np.fft.rfftfreq(nfft, dt)
    df = f[1] - f[0]
    ifreqs = [int(np.random.normal(f, sigma)/df)
              for f, sigma in zip(f0, sigmaf)]
    amps = [np.random.normal(a, sigma) for a, sigma in zip(a0, sigmaa)]
    phis = [np.random.normal(phi, sigma) for phi, sigma in zip(phi0, sigmaphi)]

    # input signal in frequency domain
    X = np.zeros(nfft//2+1, dtype='complex128')
    X[ifreqs] = np.array(amps).squeeze() * \
                np.exp(1j * np.deg2rad(np.array(phis))).squeeze()

    # input signal in time domain
    FFTop = pylops.signalprocessing.FFT(nt, nfft=nfft, real=True)
    x = FFTop.H*X
    return x

# Priors
nreals = 100
f0 = [5, 3, 8]
sigmaf = [0.5, 1., 0.6]
a0 = [1., 1., 1.]
sigmaa = [0.1, 0.5, 0.6]
phi0 = [-90., 0., 0.]
sigmaphi = [0.1, 0.2, 0.4]
sigmad = 1e-2

# Prior models
nt = 200
nfft = 2**11
dt = 0.004
t = np.arange(nt)*dt
xs = \
    np.array([prior_realization(f0, a0, phi0, sigmaf, sigmaa, sigmaphi, dt, nt, nfft) for _ in range(nreals)])

# True model (taken as one possible realization)
x = prior_realization(f0, a0, phi0, [0, 0, 0], [0, 0, 0], [0, 0, 0], dt, nt, nfft)


x0 = np.average(xs, axis=0)
Cm = ((xs - x0).t @ (xs - x0)) / nreals

N = 30 # lenght of decorrelation
diags = np.array([Cm[i, i-N:i+N+1] for i in range(N, nt-N)])
diag_ave = np.average(diags, axis=0)
diag_ave *= np.hamming(2*N+1) # add a taper at the end to avoid edge effects

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(t, xs.T, 'r', lw=1)
ax.plot(t, x0, 'g', lw=4)
ax.plot(t, x, 'k', lw=4)
ax.set_title('Prior realizations and mean')
ax.set_xlim(0, 0.8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
im = ax1.imshow(Cm, interpolation='nearest', cmap='seismic',
                extent=(t[0], t[-1], t[-1], t[0]))
ax1.set_title(r"$\mathbf{C}_m^{prior}$")
ax1.axis('tight')
ax2.plot(np.arange(-N, N + 1) * dt, diags.T, '--r', lw=1)
ax2.plot(np.arange(-N, N + 1) * dt, diag_ave, 'k', lw=4)
ax2.set_title("Averaged covariance 'filter'")
plt.show()


# Sampling operator
perc_subsampling = 0.2
ntsub = int(np.round(nt*perc_subsampling))
iava = np.sort(np.random.permutation(np.arange(nt))[:ntsub])
iava[-1] = nt-1 # assume we have the last sample to avoid instability
Rop = pylops.Restriction(nt, iava, dtype='float64')

# Covariance operators
Cm_op = \
    pylops.signalprocessing.Convolve1D(nt, diag_ave, offset=N)
Cd_op = sigmad**2 * pylops.Identity(ntsub)

n = np.random.normal(0, sigmad, nt)
y = Rop * x
yn = Rop * (x + n)
ymask = Rop.mask(x)
ynmask = Rop.mask(x + n)


xbayes = x0 + Cm_op * Rop.H * (lsqr(Rop * Cm_op * Rop.H + Cd_op,
                                    yn - Rop*x0, iter_lim=400)[0])

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(t, x, 'k', lw=6, label='true')
ax.plot(t, ymask, '.k', ms=25, label='available samples')
ax.plot(t, ynmask, '.r', ms=25, label='available noisy samples')
ax.plot(t, xbayes, 'r', lw=3, label='bayesian inverse')
ax.legend()
ax.set_title('Signal')
ax.set_xlim(0, 0.8)
plt.show()