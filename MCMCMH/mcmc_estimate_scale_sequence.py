import numpy as np
import matplotlib.pyplot as plt
from KF.generate_trajectory import *

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

points_number = 500
points_reference = 5.0 * generate_trajectory(num_samples=500, noise_mult=0.005)
points_scaled = np.copy(points_reference)
points_scaled[0] *= 0.6

plt.plot(points_reference[0], points_reference[1])
plt.plot(points_scaled[0], points_scaled[1])
plt.show()

transition_model = lambda x: [np.random.normal(x[0], 0.1, (1,)), x[1]]

# Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        return (accept < (np.exp(x_new - x)))

def prior(x):
    # x[0] = mu, x[1]=sigma (new or current)
    # returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    # returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    # It makes the new sigma infinitely unlikely.
    if (x[1] <= 0):
        return 0
    return 1

# Computes the likelihood of the data given a sigma (new or current) according to equation (2)
def log_likelihood(x, data):
    # x[0]=mu, x[1]=sigma (new or current)
    # data = the observation
    return np.sum(-np.log(x[1] * np.sqrt(2 * np.pi)) - ((data - x[0]) ** 2) / (2 * x[1] ** 2))


def calc_errors(points, points_scaled, x):
    points_scaled__ = np.copy(points)
    points_scaled__[0] *= x[0]
    return np.linalg.norm(points_scaled - points_scaled__, axis=0)


x = [1.0, 0.05]
iterations = 100
accepted = []
rejected = []

for i in range(iterations):
    x_new = transition_model(x)
    data = calc_errors(points_reference, points_scaled, x)
    x_lik = log_likelihood(x, data)
    x_new_lik = log_likelihood(x_new, data)
    # if (acceptance_rule(x_lik + np.log(prior(x)), x_new_lik + np.log(prior(x_new)))):
    #     x = x_new
    #     accepted.append(x_new)
    # else:
    #     rejected.append(x_new)
