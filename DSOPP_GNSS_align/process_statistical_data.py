import numpy as np
import pandas as pd
import os

path_root = '/home/morzh/work/DSOPP_tests_data_temp/proto_track/statistics_008'
filename_base = 'alignment_statistics'
filename_extension = '.stats'
files = [f for f in os.listdir(path_root) if os.path.isfile(os.path.join(path_root, f)) if f.endswith(filename_extension) and f.__contains__(filename_base)]

altitude_bias_ratio = np.empty(0)
max_altitude_bias_ratio = np.empty(0)
mean_altitude_bias_ratio = np.empty(0)
min_altitude_bias_ratio = np.empty(0)

tangent_bias_ratio = np.empty(0)
max_tangent_bias_ratio = np.empty(0)
mean_tangent_bias_ratio = np.empty(0)
min_tangent_bias_ratio = np.empty(0)

mean_altitude_error = np.empty(0)
mean_tangent_error = np.empty(0)
deviation_altitude_error = np.empty(0)
deviation_tangent_error = np.empty(0)

for file in files:
    filepath = os.path.join(path_root, file)
    # print(filepath)
    statistics_data = pd.read_csv(filepath)
    altitude_bias_ratio = np.append(altitude_bias_ratio, statistics_data.altitude_bias_ratio)
    max_altitude_bias_ratio = np.append(max_altitude_bias_ratio, statistics_data.max_altitude_bias_ratio)
    mean_altitude_bias_ratio = np.append(mean_altitude_bias_ratio, statistics_data.mean_altitude_bias_ratio)
    min_altitude_bias_ratio = np.append(min_altitude_bias_ratio, statistics_data.min_altitude_bias_ratio)

    tangent_bias_ratio = np.append(tangent_bias_ratio, statistics_data.tangent_bias_ratio)
    max_tangent_bias_ratio = np.append(max_tangent_bias_ratio, statistics_data.max_tangent_bias_ratio)
    mean_tangent_bias_ratio = np.append(mean_tangent_bias_ratio, statistics_data.mean_tangent_bias_ratio)
    min_tangent_bias_ratio = np.append(min_tangent_bias_ratio, statistics_data.min_tangent_bias_ratio)

    mean_altitude_error = np.append(mean_altitude_error, statistics_data.mean_altitude_error)
    mean_tangent_error = np.append(mean_tangent_error, statistics_data.mean_tangent_error)
    deviation_altitude_error = np.append(deviation_altitude_error, statistics_data.deviation_altitude_error)
    deviation_tangent_error = np.append(deviation_tangent_error, statistics_data.deviation_tangent_error)


print('average altitude bias ratio', np.mean(altitude_bias_ratio))
print('average maximum altitude bias ratio', np.mean(max_altitude_bias_ratio))
print('average mean of altitude bias ratio', np.mean(mean_altitude_bias_ratio))
print('average min of altitude bias ratio', np.mean(min_altitude_bias_ratio))
print(' ')
print('average tangent bias ratio', np.mean(tangent_bias_ratio))
print('average maximum tangent bias ratio', np.mean(max_tangent_bias_ratio))
print('average mean tangent bias ratio', np.mean(mean_tangent_bias_ratio))
print('average min tangent bias ratio', np.mean(min_tangent_bias_ratio))
print(' ')
print('average mean altitude error', np.mean(mean_altitude_error))
print('average mean tangent error', np.mean(mean_tangent_error))
print('average deviation altitude error', np.mean(deviation_altitude_error))
print('average deviation tangent error', np.mean(deviation_tangent_error))




