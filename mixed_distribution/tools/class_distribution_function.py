import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve


class DistributionFunction:
    # func with following props
    # continuous, defined on an interval of form
    # [a, b]
    # with integral == 1
    def __init__(self, func, interval):
        self.a, self.b = interval

        # vectorize dist function
        self.func = np.vectorize(func)

        # create table of [(xi, pi)]
        self.FX_hat = []
        for x in np.linspace(self.a, self.b, 50):
            self.FX_hat.append((x, self.cumulative_dist_function(x)))

        return

    def __call__(self, x):
        return self.func(x)

    # cumulative distribution function F
    def cumulative_dist_function(self, x):
        a, b = self.a, self.b

        # case x lies on the left side of domain
        # of definition
        if x < a:
            return 0

        # case x lies on the left side of domain
        # of definition
        if x >= b:
            return 1

        return quad(lambda u: self(u), a, x)[0]

    # for a given prob p0 find corresponding y
    # for which F(y) = p0
    def map_dist_uniform(self, p0):
        # equation to solve
        equation_func = lambda x: self.cumulative_dist_function(x) - p0

        # starting point of solver
        # take mid point
        FX_hat, n = self.FX_hat, len(self.FX_hat)
        i, j, = generalized_binary_search(arr_obj=FX_hat, i=0, j=n, target=p0, metric=lambda x: x[1])

        x0 = (FX_hat[i][0] + FX_hat[j][0]) / 2

        # solve
        y = fsolve(func=equation_func, x0=np.array([x0]), full_output=True)

        return y[0][0]

    def generate_samples(self, size=1):
        # sample from a uniform [0, 1]
        arr_p0 = np.random.rand(size)

        # corresponding mappings of p0
        Y = []

        for p0 in arr_p0:
            y = self.map_dist_uniform(p0)

            Y.append(y)
            # print(y)
        return Y


# binary search performed on object
# objective is find i, j such that metric(ith-object) <= target < metric(jth-object)
# assume len(arr_obj) > 1
# assume that: metric(1st-object) <= target < metric(nth-object)
def generalized_binary_search(arr_obj, i, j, target, metric=lambda x: x):
    if j - i == 1:
        return i, j

    # index of the middle point
    mid = int((i + j) / 2)
    current_val = metric(arr_obj[mid])

    if current_val == target:
        return mid, mid + 1

    if current_val > target:
        return generalized_binary_search(arr_obj, i, mid, target, metric)

    return generalized_binary_search(arr_obj, mid, j, target, metric)


# # example Chauchy distribution
# interval = [-10, 10]
# dis_func = lambda u: (1 / np.pi) * 1 / (1 + u ** 2) * RecFunction(*interval)(u)
#
# dist = DistributionFunction(func=dis_func, interval=interval)
#
# # plot of distribution
# x = np.linspace(*interval, 200)
# plt.plot(x, dist(x))
#
# plt.show()
#
# # plot of cumulative distribution function
# probs = []
#
# for val in x:
#     p = dist.cumulative_dist_function(val)
#
#     probs.append(p)
#
# plt.plot(x, probs)
# plt.show()
#
# # plot mapping uniform distribution
#
# # remove borders
# probs = np.linspace(0, 1, 100)[1:-1]
# y = []
#
# for p0 in probs:
#     y0 = dist.map_dist_uniform(p0)
#
#     y.append(y0)
#
# plt.plot(probs, y)
#
# plt.show()
