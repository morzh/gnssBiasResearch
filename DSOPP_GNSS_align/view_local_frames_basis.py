import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

main_path = '/home/morzh/work/DSOPP_tests_data_temp/local_frames_coordinates_frame'
ecef_frame_filebase = 'ecef_frame'
local_frame_filebase = 'local_frames_basis'
number_elements = 250

for idx in range(number_elements):
    if idx % 10 != 0:
        continue
    filepath_ecef_poses = os.path.join(main_path, ecef_frame_filebase+'.'+str(idx)+'.csv')
    filepath_local_basis = os.path.join(main_path, local_frame_filebase+'.'+str(idx)+'.csv')

    ecef_poses = pd.read_csv(filepath_ecef_poses)
    local_frame_basis = pd.read_csv(filepath_local_basis)
    '''
    ecef_poses = np.vstack((ecef_poses['x'].array.to_numpy().reshape(1, -1),
                            ecef_poses['y'].array.to_numpy().reshape(1, -1),
                            ecef_poses['z'].array.to_numpy().reshape(1, -1)))
    '''
    x = ecef_poses['x']
    y = ecef_poses['y']
    z = ecef_poses['z']

    u_1 = local_frame_basis['x1']
    u_2 = local_frame_basis['y1']
    u_3 = local_frame_basis['z1']

    v_1 = local_frame_basis['x2']
    v_2 = local_frame_basis['y2']
    v_3 = local_frame_basis['z2']

    w_1 = local_frame_basis['x3']
    w_2 = local_frame_basis['y3']
    w_3 = local_frame_basis['z3']

    start = 0
    end = 2685

    fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # plt.plot(x, y)
    plt.quiver(x[start:end], y[start:end], u_1[start:end], u_2[start:end], scale=100, width=.0005, headlength=0, color='b')
    plt.quiver(x[start:end], y[start:end], v_1[start:end], v_2[start:end], scale=100, width=.0005, headlength=0)
    # plt.quiver(x, y, w_1, w_2, scale=0.60, width=.005, headlength=10)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
