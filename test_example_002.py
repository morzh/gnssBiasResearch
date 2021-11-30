import cvxpy as cp

x = cp.Variable()
y = cp.Variable(pos=True)
objective_fn = -cp.sqrt(x) / y
problem = cp.Problem(cp.Minimize(objective_fn), [cp.exp(x) <= y])
print("Is problem DQCP?: ", problem.is_dqcp())
print("Is problem DCP?: ", problem.is_dcp())
problem.solve(qcp=True)
assert problem.is_dqcp()
print("Optimal value: ", problem.value)
print("x: ", x.value)
print("y: ", y.value)