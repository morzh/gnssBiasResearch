import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
# from cvxpy_atoms.total_variation_1d import *
from cvxpy.reductions.cvx_attr2constr import *
from cvxpy.reductions.dcp2cone.dcp2cone import *
from cvxpy.reductions.cvx_attr2constr import *
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.promote import promote

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
elements_number = 3
dimensions = 2
nnz = 1

# prm = np.random.permutation(dimensions + 1)
betaTrue = np.zeros(elements_number)
betaTrue[1] = 2.25

# X = np.random.randn(elements_number, dimensions)
# X = np.insert(X, 0, 1, axis=1)
noise = 0.001 * np.random.randn(elements_number)
y = betaTrue #+ noise
# y = X @ betaTrue + noise
lmbda = 0.1

print(y)


beta = cp.Variable(dimensions + 1)
functions = [cp.sum_squares(beta - y), lmbda * cp.sum(cp.norm1(beta, axis=0))]
problem = cp.Problem(cp.Minimize(sum(functions)))
original_data, original_chain, original_inverse_data = problem.get_problem_data(cp.ECOS)

'''
promote_2_3 = promote(2.0, (3,))

dcp2cone = Dcp2Cone()
dcp2cone_problem = dcp2cone.apply(problem)[0]
dcp2cone_data, dcp2cone_chain, dcp2cone_inverse_data = dcp2cone_problem.get_problem_data(cp.ECOS)

cvxAttr2Constr = CvxAttr2Constr()
cvxAttr2Constr_problem = cvxAttr2Constr.apply(dcp2cone_problem)[0]
cvxAttr2Constr_data, cvxAttr2Constr_chain, cvxAttr2Constr_inverse_data = cvxAttr2Constr_problem.get_problem_data(cp.ECOS)
'''
solution = problem.solve(solver=cp.ECOS, warm_start=True)
# attr2const = CvxAttr2Constr()
# attr2const_problem = attr2const.apply(problem)

print('sdfsdfsd')
