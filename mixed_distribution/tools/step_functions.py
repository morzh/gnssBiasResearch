import numpy as np
import matplotlib.pyplot as plt


class Heaviside:
    def __init__(self, x0=0):
        self.x0 = x0

        self.vectorized_function = np.vectorize(self.f)
        return

    def f(self, x):
        if x >= self.x0:
            return 1
        else:
            return 0

    def __call__(self, x):
        return self.vectorized_function(x)


class RecFunction:
    def __init__(self, a, b):
        self.a, self.b = a, b

        return

    def __call__(self, x):
        a, b, = self.a, self.b

        return Heaviside(a)(x) - Heaviside(b)(x)


# # example heaviside
# t = np.linspace(-5, 5, 100)
#
# x0 = 1
# h_function = Heaviside(x0=x0)
#
# plt.plot(t, h_function(t))
# plt.show()


# # example rec function
# t = np.linspace(-5, 5, 100)
#
# a, b = 1, 3
# rec_function = RecFunction(a, b)
#
# plt.plot(t, rec_function(t))
# plt.show()
