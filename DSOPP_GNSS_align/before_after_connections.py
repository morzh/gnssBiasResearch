import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

before_csv = pd.read_csv('/home/morzh/work/DSOPP_data_output/before_connections.csv')
after_csv = pd.read_csv('/home/morzh/work/DSOPP_data_output/after_connections.csv')

fig = plt.figure(figsize=(20, 10))
ax = fig.gca(projection='3d')
ax.plot(before_csv.x_1, before_csv.y_1, before_csv.z_1_subproblem_1, linewidth=6)
ax.plot(after_csv.x_1, after_csv.y_1, after_csv.z_1_subproblem_1)
plt.show()