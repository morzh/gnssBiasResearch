from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt

# Problem data.
m = 100
n = 75
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m,)
gamma = 1.05

xs = np.linspace(1, m, m)

rho = 1.0
x = Variable(n)
funcs = [sum_squares(A@x - b), gamma*norm(x, 2)]
prob = Problem(Minimize(sum(funcs)))
result = prob.solve()

plt.figure(figsize=(20, 10))
# plt.plot(x, sine, color='green')
plt.plot(xs, b, color='red')
plt.plot(xs, A@x.value, color='green')
# plt.plot(x, bias, color='blue')
plt.tight_layout()
plt.show()