import numpy as np
from scipy.spatial.transform import Rotation as Rotation

r = 0.65
scale = 0.5
noise_mult = 0.2

R = Rotation.from_euler('z', r, degrees=False)
R = R.as_dcm()
sR = scale*R

