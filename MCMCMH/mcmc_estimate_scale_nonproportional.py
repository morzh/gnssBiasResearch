import numpy as np
import matplotlib.pyplot as plt
from KF.generate_trajectory import *

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

points_number = 500
points = 5.0*generate_trajectory(num_samples=500, noise_mult=0.005)
points_scaled = np.copy(points)
points_scaled[0] *= 0.56
points_scaled[1] *= 0.38

plt.plot(points[0], points[1])
plt.plot(points_scaled[0], points_scaled[1])
plt.show()


# transition_model = lambda x: np.array([float(np.random.multivariate_normal(mean, cov, 1)), x[1]])
def transition_model(mean, cov):
    sample = np.random.multivariate_normal(mean.flatten(), cov, 1)
    return sample.reshape(2, 1)

# Defines whether to accept or reject the new sample
def acceptance_rule(x, x_new):
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        return (accept < (np.exp(x_new - x)))

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def prior(x):
    # x[0] = mu, x[1]=sigma (new or current)
    # returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    # returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    # It makes the new sigma infinitely unlikely.
    if is_pos_def(x):
        return 1
    else:
        return 0


def log_likelihood(mean, cov, points_1, points_2):
    data = calc_errors(points_1, points_2, mean)
    sum = 0.0
    for idx in range(points_1.shape[1]):
        sum -= 0.5*np.matmul(data[:, idx].T, np.linalg.inv(cov)).dot(data[:, idx]) + np.log(np.sqrt(2 * np.pi)**2 * np.linalg.det(cov))
    return sum
    # return np.sum(-np.log(np.sqrt(2 * np.pi)**2 * np.linalg.det(cov)) - 0.5 * np.matmul(data.T, np.linalg.inv(cov)).dot(data))
    # return np.sum(-np.log(x[1] * np.sqrt(2 * np.pi)) - ((data - x[0]) ** 2) / (2 * x[1] ** 2))


def calc_errors(points, points_scaled, mean):
    points_scaled__ = np.copy(points)
    points_scaled__[0] *= mean[0]
    points_scaled__[1] *= mean[1]
    return points_scaled - points_scaled__


mean = np.array([1.0, 1.0]).reshape(2, 1)
cov = 0.25*np.eye(2)

iterations = int(3e4)

accepted = np.empty((2, 0))
rejected = np.empty((2, 0))

print('starting MCMC....')
for i in range(iterations):
    if i % 1e3 == 0 and i > 0:
        print('iteration', i, 'accepted', accepted.shape[1], 'samples, rejected', rejected.shape[1], 'samples, mean is', np.mean(accepted, axis=1))

    mean_new = transition_model(mean, cov)
    mean_lik = log_likelihood(mean, cov, points, points_scaled)
    mean_new_lik = log_likelihood(mean_new, cov, points, points_scaled)

    if acceptance_rule(mean_lik, mean_new_lik):
    # if acceptance_rule(mean_lik + np.log(prior(mean)), mean_new_lik + np.log(prior(mean_new))):
        mean = mean_new
        accepted = np.hstack((accepted, mean_new))
    else:
        rejected = np.hstack((rejected, mean_new))


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(2, 1, 1)
ax.plot(rejected[0], 'rx', label='Rejected', alpha=0.5)
ax.plot(accepted[0], 'b.', label='Accepted', alpha=0.5)
ax.grid()
ax = fig.add_subplot(2, 1, 2)
ax.plot(rejected[1], 'rx', label='Rejected', alpha=0.5)
ax.plot(accepted[1], 'b.', label='Accepted', alpha=0.5)
ax.grid()
plt.show()

mean = np.mean(accepted[:, 3:], axis=1)
print(mean)
plt.plot(points_scaled[0], points_scaled[1])
plt.plot(points[0]*mean[0], points[1]*mean[1])
plt.show()

print(np.mean(accepted[:, 6:], axis=1))
