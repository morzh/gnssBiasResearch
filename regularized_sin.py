# pylint: disable=C0103
import numpy as np
import matplotlib.pyplot as plt
import pylops
plt.close('all')
np.random.seed(10)

# Signal creation in frequency domain
ifreqs = [41, 25, 66]
amps = [1., 1., 1.]
N = 200
nfft = 2**11
dt = 0.004
t = np.arange(N)*dt
f = np.fft.rfftfreq(nfft, dt)

FFTop = 10*pylops.signalprocessing.FFT(N, nfft=nfft, real=True)

X = np.zeros(nfft//2+1, dtype='complex128')
X[ifreqs] = amps
x = FFTop.H*X

fig, axs = plt.subplots(2, 1, figsize=(12, 8))
axs[0].plot(f, np.abs(X), 'k', LineWidth=2)
axs[0].set_xlim(0, 30)
axs[0].set_title('Data(frequency domain)')
axs[1].plot(t, x, 'k', LineWidth=2)
axs[1].set_title('Data(time domain)')
axs[1].axis('tight')
plt.show()


# subsampling locations
perc_subsampling = 0.2
Nsub = int(np.round(N*perc_subsampling))

iava = np.sort(np.random.permutation(np.arange(N))[:Nsub])

# Create restriction operator
Rop = pylops.Restriction(N, iava, dtype='float64')

y = Rop*x
ymask = Rop.mask(x)

# Visualize data
fig = plt.figure(figsize=(12, 4))
plt.plot(t, x, 'k', lw=3)
plt.plot(t, x, '.k', ms=20, label='all samples')
plt.plot(t, ymask, '.g', ms=15, label='available samples')
plt.legend()
plt.title('Data restriction')
plt.show()

xinv = pylops.optimization.leastsquares.RegularizedInversion(Rop, [], y, **dict(damp=0, iter_lim=10, show=1))
xinv_fromx0 = pylops.optimization.leastsquares.RegularizedInversion(Rop, [], y, x0=np.ones(N), **dict(damp=0, iter_lim=10, show=0))
xne = pylops.optimization.leastsquares.NormalEquationsInversion(Rop, [], y)

fig = plt.figure(figsize=(12, 4))
plt.plot(t, x, 'k', lw=2, label='original')
plt.plot(t, xinv, 'b', ms=10, label='inversion')
plt.plot(t, xinv_fromx0, '--r', ms=10, label='inversion from x0')
plt.plot(t, xne, '--g', ms=10, label='normal equations')
plt.legend()
plt.title('Data reconstruction without regularization')
plt.show()

##########################################################################33

# Create regularization operator
D2op = pylops.SecondDerivative(N, dims=None, dtype='float64')

# Regularized inversion
epsR = np.sqrt(0.1)
epsI = np.sqrt(1e-4)

xne = \
    pylops.optimization.leastsquares.NormalEquationsInversion(Rop, [D2op], y,
                                                              epsI=epsI,
                                                              epsRs=[epsR],
                                                              returninfo=False,
                                                              **dict(maxiter=50))

ND2op = pylops.MatrixMult((D2op.H * D2op).tosparse()) # mimic fast D^T D

xne1 = \
    pylops.optimization.leastsquares.NormalEquationsInversion(Rop, [], y,
                                                              NRegs=[ND2op],
                                                              epsI=epsI,
                                                              epsNRs=[epsR],
                                                              returninfo=False,
                                                              **dict(maxiter=50))

xreg = \
    pylops.optimization.leastsquares.RegularizedInversion(Rop, [D2op], y,
                                                          epsRs=[np.sqrt(0.1)],
                                                          returninfo=False,
                                                          **dict(damp=np.sqrt(1e-4),
                                                                 iter_lim=50,
                                                                 show=0))

# Create regularization operator
Sop = pylops.Smoothing1D(nsmooth=11, dims=[N], dtype='float64')

# Invert for interpolated signal
xprec = \
    pylops.optimization.leastsquares.PreconditionedInversion(Rop, Sop, y,
                                                             returninfo=False,
                                                             **dict(damp=np.sqrt(1e-9),
                                                                    iter_lim=20,
                                                                    show=0))


# sphinx_gallery_thumbnail_number=4
fig = plt.figure(figsize=(12, 4))
plt.plot(t[iava], y, '.k', ms=20, label='available samples')
plt.plot(t, x, 'k', lw=3, label='original')
plt.plot(t, xne, 'b', lw=3, label='normal equations')
plt.plot(t, xne1, '--c', lw=3, label='normal equations (with direct D^T D)')
plt.plot(t, xreg, '-.r', lw=3, label='regularized')
plt.plot(t, xprec, '--g', lw=3, label='preconditioned equations')
plt.legend()
plt.title('Data reconstruction with regularization')

subax = fig.add_axes([0.7, 0.2, 0.15, 0.6])
subax.plot(t[iava], y, '.k', ms=20)
subax.plot(t, x, 'k', lw=3)
subax.plot(t, xne, 'b', lw=3)
subax.plot(t, xne1, '--c', lw=3)
subax.plot(t, xreg, '-.r', lw=3)
subax.plot(t, xprec, '--g', lw=3)
subax.set_xlim(0.05, 0.3)
plt.show()



pista, niteri, costi = \
    pylops.optimization.sparsity.ISTA(Rop*FFTop.H, y, niter=1000,
                                      eps=0.1, tol=1e-7, returninfo=True)
xista = FFTop.H*pista

pfista, niterf, costf = \
    pylops.optimization.sparsity.FISTA(Rop*FFTop.H, y, niter=1000,
                                       eps=0.1, tol=1e-7, returninfo=True)
xfista = FFTop.H*pfista

fig, axs = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Data reconstruction with sparsity', fontsize=14,
             fontweight='bold', y=0.9)
axs[0].plot(f, np.abs(X), 'k', lw=3)
axs[0].plot(f, np.abs(pista), '--r', lw=3)
axs[0].plot(f, np.abs(pfista), '--g', lw=3)
axs[0].set_xlim(0, 30)
axs[0].set_title('Frequency domain')
axs[1].plot(t[iava], y, '.k', ms=20, label='available samples')
axs[1].plot(t, x, 'k', lw=3, label='original')
axs[1].plot(t, xista, '--r', lw=3, label='ISTA')
axs[1].plot(t, xfista, '--g', lw=3, label='FISTA')
axs[1].set_title('Time domain')
axs[1].axis('tight')
axs[1].legend()
plt.tight_layout()
plt.subplots_adjust(top=0.8)

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.semilogy(costi, 'r', lw=2, label='ISTA')
ax.semilogy(costf, 'g', lw=2, label='FISTA')
ax.set_title('Cost functions', size=15, fontweight='bold')
ax.set_xlabel('Iteration')
ax.legend()
ax.grid(True)
plt.tight_layout()


xspgl1, pspgl1, info = \
    pylops.optimization.sparsity.SPGL1(Rop, y, FFTop, tau=3, iter_lim=200)

fig, axs = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Data reconstruction with SPGL1', fontsize=14,
             fontweight='bold', y=0.9)
axs[0].plot(f, np.abs(X), 'k', lw=3)
axs[0].plot(f, np.abs(pspgl1), '--m', lw=3)
axs[0].set_xlim(0, 30)
axs[0].set_title('Frequency domain')
axs[1].plot(t[iava], y, '.k', ms=20, label='available samples')
axs[1].plot(t, x, 'k', lw=3, label='original')
axs[1].plot(t, xspgl1, '--m', lw=3, label='SPGL1')
axs[1].set_title('Time domain')
axs[1].axis('tight')
axs[1].legend()
plt.tight_layout()
plt.subplots_adjust(top=0.8)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.semilogy(info['rnorm2'], 'k', lw=2, label='ISTA')
ax.set_title('Cost functions', size=15, fontweight='bold')
ax.set_xlabel('Iteration')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()