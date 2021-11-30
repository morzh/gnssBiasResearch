from KF.generate_trajectory import *
import fdasrsf

N_iter = 600

[points_1, points_2] = generate_trajectories(num_samples=N_iter, trajectory_noise=(0.125, 0.1), noise_mult=(0.007, 0.009), scale=5.0, random_points_num=150)

points_1 = np.array(points_1)
points_2_src = np.array(points_2)

theta = 0.9
t = np.array([2, 3])
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

plt.plot(points_1[0], points_1[1])
plt.plot(points_2_src[0], points_2_src[1])
plt.show()

points_2 = np.matmul(R, points_2_src) + t.reshape(2, 1)

plt.plot(points_1[0], points_1[1])
plt.plot(points_2[0], points_2[1])
plt.show()

curves = np.dstack((points_1, points_2))

fda_curve = fdasrsf.curve_stats.fdacurve(curves, N=N_iter)
fda_curve.karcher_mean()

print(fda_curve.beta_mean)

plt.plot(fda_curve.beta_mean[0], fda_curve.beta_mean[1])
plt.plot(points_1[0], points_1[1])
plt.plot(points_2[0], points_2[1])
plt.show()

print('aligning curve ....')
fda_curve.srvf_align()
print(type(fda_curve.betan))
print(fda_curve.betan.shape)


plt.figure(figsize=(20, 10))
plt.plot(fda_curve.beta_mean[0], fda_curve.beta_mean[1])
plt.plot(points_1[0], points_1[1])
plt.plot(points_2_src[0], points_2_src[1])
for idx in range(fda_curve.betan.shape[0]):
    plt.plot(fda_curve.betan[0, :, idx], fda_curve.betan[1, :, idx])
plt.show()
