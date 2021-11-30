import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import sophus
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

d = 2
N = 1000
r = 0.65

linspace = np.linspace(0.0, float(N-1), N)
x = np.zeros((d, N))
x[0] = linspace

values = []
angles = np.linspace(-1.0, 1.0, 1000)

for angle in angles:
    R = Rotation.from_euler('z', angle, degrees=False)
    R = R.as_matrix()[0:2, 0:2]
    y = R @ x
    value = np.sum(np.linalg.norm(y-x, axis=0)**2)
    values.append(value)

plt.plot(linspace, values)
plt.show()
