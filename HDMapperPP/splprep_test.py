import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev

phi = np.linspace(0, 2.*np.pi, 40)
r = 0.5 + np.cos(phi)         # polar coords
x, y = r * np.cos(phi), r * np.sin(phi)    # convert to cartesian

points = np.vstack((x, y))

tck, u = splprep(points)
new_points = splev(u, tck)