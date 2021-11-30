import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
# from cvxpy_atoms.my_cos import *

epsilon = 1
x = cp.Variable()
func = cp.my_cos(x)
objective = cp.Minimize(func)

'''
vals = []
xs = np.linspace(-1, 1, 150)
for val in xs:
    x.value = val
    vals.append(np.float64(func.value))
plt.plot(xs, vals)
plt.show()
'''

x.value = 0.3
problem = cp.Problem(objective, [cp.abs(x) <= epsilon])
print("Is problem DQCP?: ", problem.is_dqcp())
print("Is problem DCP?: ", problem.is_dcp())
problem.solve(qcp=True)
# problem.solve(qcp=True, warm_start=True)
print('optimal value is:', x.value)
