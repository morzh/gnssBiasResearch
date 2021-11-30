import os
import pandas as pd
from scipy.interpolate import splprep, splev
import fdasrsf
from HDMapperPP.hdmapper_utils import *
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import pandas as pd

def save_houdini_csv(P, path2save):
    import pandas as pd
    df = pd.DataFrame(P)
    df.columns = ['P[x]', 'P[y]', 'P[z]']
    df.to_csv(path2save, index=False)



def save_houdini_accuracy_csv(P, accuracy, path2save):
    df = pd.DataFrame(np.hstack((P, accuracy)))
    df.columns = ['P[x]', 'P[y]', 'P[z]', 'acc[x]', 'acc[y]', 'acc[z]']
    df.to_csv(path2save, index=False)


P = np.array([[0.1, -0.2, 0.3], [1, 2.2, 2.3]])
accuracy = np.array([[1, 2, 3], [4, 5, 6]])

save_houdini_accuracy_csv(P, accuracy, '/home/morzh/work/hdmapper_data/test.csv')