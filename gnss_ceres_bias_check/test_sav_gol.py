from scipy.signal import savgol_filter
import numpy as np

data = np.array([1,2,1,4,5,2,7,8,3,10,11,12,13,14,15,16,17])

bias_smoothed = savgol_filter(data, 11, 3)

print(bias_smoothed)