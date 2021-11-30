from KF.generate_trajectory import *

bias_smoothed = get_smoothed_piecewise_bias(number_iterations=100, jumps_number=(1, 10), mult=0.05, smooth_factor=(25, 15), savgol_params=(35, 3))

print(bias_smoothed)