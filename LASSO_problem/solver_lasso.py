import numpy as np
import matplotlib.pyplot as plt

def clip(beta, alpha):
    clipped = np.minimum(beta, alpha)
    clipped = np.maximum(clipped, -alpha)
    return clipped

def proxL1Norm(betaHat, alpha, penalizeAll=True):
    out = betaHat - clip(betaHat, alpha)
    if not penalizeAll:
        out[0] = betaHat[0]

    return out


np.random.seed(19)
N = 40
d = 200
nnz = 3

prm = np.random.permutation(d+1)
betaTrue = np.zeros(d+1)
betaTrue[prm[0:nnz]] = 5*np.random.randn(nnz)

X = np.random.randn(N, d)
X = np.insert(X, 0, 1, axis=1)
noise = 0.001 * np.random.randn(N)
y = X @ betaTrue + noise

plt.stem(betaTrue)
plt.title('Beta ground truth')
plt.show()


def solve_lasso_prox_grad(x, y, lmbda):
    maxIter = 300
    alpha = 0.005

    beta = np.zeros(d+1)
    costFunVal = np.zeros(maxIter)

    for t in range(maxIter):
        grad = X.T @ (X @ beta - y)
        beta = proxL1Norm(beta - alpha * grad, alpha * lmbda)
        costFunVal[t] = 0.5 * np.linalg.norm(X @ beta - y)**2 + lmbda * np.sum(np.abs(beta))
        print('iteration', t, 'cost value:', costFunVal[t])
    return beta, costFunVal


lmbda = 10
beta, cost_vals = solve_lasso_prox_grad(X, y, lmbda)

plt.figure()
plt.stem(beta, markerfmt='C1o')
plt.stem(betaTrue)
plt.show()

plt.semilogy(cost_vals)
plt.show()