import numpy as np


def product_laplace_mu_inf(x, mu, b1, b2):
    return np.exp(-x/b1 - (x-mu)/b2)

def product_laplace_zero_mu(x, mu, b1, b2):
    return np.exp(-x/b1 - (mu-x)/b2)

def product_laplace_minus_inf_zero(x, mu, b1, b2):
    return np.exp(x/b1 - (mu-x)/b2)

def product_laplace_minus_inf_plus_inf(x, mu, b1, b2):
    return np.exp(-np.abs(x)/b1 - np.abs(x-mu)/b2)


mu = 25
b1 = 1.1
b2 = 1.5
'''
# mu -- inf
x_1 = np.linspace(mu, mu + 1e5, 400 * int(1e5) + 1)
y_1 = product_laplace_mu_inf(x_1, mu, b1, b2)

print(0.0025 * np.sum(y_1))
print((b1*b2)/(b1+b2) * np.exp(-mu/b1))
print('---------------------------------')
'''
'''
# 0 -- mu
x_2 = np.linspace(0, mu, int(1e5) + 1)
y_2 = product_laplace_zero_mu(x_2, mu, b1, b2)
step = mu / int(1e5)

print(step * np.sum(y_2))
print((b1*b2)/(b2-b1) * (np.exp(-mu/b1) - np.exp(-mu/b2)))
print('---------------------------------')
'''
'''
#-inf -- 0
x_3 = np.linspace(-1e5, 0, 400 * int(1e5) + 1)
y_3 = product_laplace_minus_inf_zero(x_3, mu, b1, b2)

print(0.0025 * np.sum(y_3))
print((b1*b2)/(b1+b2) * np.exp(-mu/b2))
'''
'''
#-inf -- inf
x__ = np.linspace(-1e5, 1e5, 400 * int(2*1e5) + 1)
y__ = product_laplace_minus_inf_plus_inf(x__, mu, b1, b2)

print(0.0025 * np.sum(y__))
print(b1*b2/(b1+b2)*(np.exp(-mu/b1) + np.exp(-mu/b2)) - b1*b2/(b2-b1)*(np.exp(-mu/b1) - np.exp(-mu/b2)))
'''

print(b1*b2/(b1+b2)*(np.exp(-mu/b1) + np.exp(-mu/b2)) - b1*b2/(b2-b1)*(np.exp(-mu/b1) - np.exp(-mu/b2)))
