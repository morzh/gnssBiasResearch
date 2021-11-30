import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from KF.KF_utils import *

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

'''
multisensors fusion using Kalman filter
'''

