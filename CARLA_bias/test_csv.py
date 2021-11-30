import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



path_root = '/home/morzh/work/ceres_admm/0002'
filename_base = 'admm_gnss_ofometry_'
show_plot = True
files = [f for f in os.listdir(path_root) if os.path.isfile(os.path.join(path_root, f)) if f.endswith('csv') and f.__contains__(filename_base)]
num_files = len(files)
for idx in range(num_files):
    filename = filename_base+str(idx)+'.csv'
    data = pd.read_csv(os.path.join(path_root, filename))
    contain_nan = data.isnull().values.any()

    if show_plot and not contain_nan:
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot(data['ENUx'], data['ENUy'], data['ENUz'])
        ax.plot(data['tx'], data['ty'], data['tz'])
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot(data['bx'], data['by'], data['bz'])
        plt.tight_layout()
        plt.show()