import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter


def get_acceleration(points, idx, dt):
    v_forward = (points[:, idx+1] - points[:, idx]) / dt
    v_backward = (points[:, idx] - points[:, idx-1]) / dt

    return (v_forward - v_backward)/dt

def get_closest_point(point, points):
    point = point.reshape(2, -1)
    min_idx = np.argmin(np.linalg.norm(points - point, axis=0))
    return points[:, min_idx], min_idx

def get_closest_point_iterative(point, vel, dt, points, num_iterations=4):
    dist = vel * dt
    dist_norm = np.linalg.norm(dist)
    point__ = point.reshape(2, -1) + dist

    for idx in range(num_iterations):
        min_idx = np.argmin(np.linalg.norm(points - point__, axis=0))
        min_dist_point = points[:, min_idx].reshape(2, 1)
        diff = min_dist_point-point
        diff_norm = np.linalg.norm(diff)
        if diff_norm < dist_norm:
            point__ = point + dist_norm * diff / diff_norm
        else:
            break

    return min_dist_point, min_idx

def get_closest_point_next(point, points):
    num_points = points.shape[1]
    point = point.reshape(2, -1)
    min_idx = np.argmin(np.linalg.norm(points - point, axis=0))
    return points[:, (min_idx+1) % num_points]

def get_trajectory(num_samples = 500, noise_mult=3e-3):
    arr = np.array([[0, 0], [2, 0.5], [2.5, 1.25], [2.6, 2.8], [1.3, 1.1]])
    x, y = zip(*arr)
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]
    f, u = interpolate.splprep([x, y], s=0, per=True)
    xint, yint = interpolate.splev(np.linspace(0, 1, num_samples), f)
    xint += noise_mult*np.random.randn(num_samples)
    yint += noise_mult*np.random.randn(num_samples)
    return np.vstack((xint, yint))


def generate_trajectory(num_samples=500, noise_mult=1.0, random_points_num=120):
    points = np.random.rand(random_points_num, 2)
    convex_hull = ConvexHull(points)
    convex_vertices_indices = convex_hull.vertices
    convex_hull_points = points[convex_vertices_indices]
    tck, u = interpolate.splprep(convex_hull_points.T, s=0, per=True)
    xi, yi = interpolate.splev(np.linspace(0, 1, num_samples), tck)
    noise = noise_mult*np.random.rand(2, num_samples)
    return np.vstack((xi, yi))+noise


def generate_trajectories(num_samples=500, trajectory_noise=(0.04, 0.03), noise_mult=(0.01, 0.006), random_points_num=120, scale=3.0, separate_noise=False):
    points = np.random.rand(random_points_num, 2)
    convex_hull = ConvexHull(points)
    convex_vertices_indices = convex_hull.vertices
    convex_hull_points_1 = points[convex_vertices_indices] + trajectory_noise[0]*np.random.rand(len(convex_vertices_indices), 2)
    convex_hull_points_2 = points[convex_vertices_indices] + trajectory_noise[1]*np.random.rand(len(convex_vertices_indices), 2)
    tck_1, u_1 = interpolate.splprep(convex_hull_points_1.T, s=0, per=True)
    tck_2, u_2 = interpolate.splprep(convex_hull_points_2.T, s=0, per=True)
    x_1, y_1 = interpolate.splev(np.linspace(0, 1, num_samples), tck_1)
    x_2, y_2 = interpolate.splev(np.linspace(0, 1, num_samples), tck_2)
    noise_1 = noise_mult[0]*np.random.rand(2, num_samples)
    noise_2 = noise_mult[1]*np.random.rand(2, num_samples)
    if separate_noise:
        return [scale * np.vstack((x_1, y_1)), scale * np.vstack((x_2, y_2)), noise_1, noise_2]
    else:
        return [scale * np.vstack((x_1, y_1)) + noise_1, scale * np.vstack((x_2, y_2)) + noise_2]


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n, n))


def get_biases(num_samples, biases_values, biases_durations_normalized):
    biases_timeline = np.array([0])
    biases_durations = num_samples*biases_durations_normalized
    x = np.empty((0, 1))
    for idx in range(biases_durations.shape[0]):
        biases_timeline = np.vstack((biases_timeline, biases_timeline[-1]+biases_durations[idx]))

    for idx in range(biases_timeline.shape[0]-1):
        x = np.vstack((x, biases_values[idx]*np.ones((round(float(biases_timeline[idx+1])) - round(float(biases_timeline[idx])), 1))))

    if x.shape[0] < num_samples:
        x = np.vstack((x, x[-1]*np.ones(num_samples-x.shape[0])))
    return x.reshape(num_samples,)

def get_smoothed_piecewise_bias(number_iterations=500, jumps_number=(3, 10), mult=0.2, smooth_factor=(25, 15), savgol_params=(35, 3)):
    bias_number_jumps = np.random.randint(jumps_number[0], jumps_number[1])
    bias_values = mult * np.random.randn(bias_number_jumps)
    bias_durations = np.abs(np.random.randn(bias_number_jumps))
    bias_durations_normalized = bias_durations / np.sum(bias_durations)
    bias = get_biases(number_iterations, bias_values, bias_durations_normalized)
    bias_smoothed = smooth(bias, smooth_factor[0])
    bias_smoothed = savgol_filter(bias_smoothed, savgol_params[0], savgol_params[1])
    bias = smooth(bias_smoothed, smooth_factor[1])

    return bias

'''
points = get_trajectory()
point = np.array([[0], [2]])

closest_point_idx = get_closest_point(point, points)

plt.plot(points[0], points[1])
plt.scatter(points[0,closest_point_idx], points[1, closest_point_idx])
plt.scatter(point[0], point[1], c='r')
plt.show()
'''