import numpy as np
import matplotlib.pyplot as plt



G_negative = np.load('/home/morzh/temp/ECOS/problem_data_negative_G.npy')
c_negative = np.load('/home/morzh/temp/ECOS/problem_data_negative_c.npy')
h_negative = np.load('/home/morzh/temp/ECOS/problem_data_negative_h.npy')

G_negpos = np.load('/home/morzh/temp/ECOS/problem_data_negpos_G.npy')
c_negpos = np.load('/home/morzh/temp/ECOS/problem_data_negpos_c.npy')
h_negpos = np.load('/home/morzh/temp/ECOS/problem_data_negpos_h.npy')

G_pos = np.load('/home/morzh/temp/ECOS/problem_data_pos_G.npy')
c_pos = np.load('/home/morzh/temp/ECOS/problem_data_pos_c.npy')
h_pos = np.load('/home/morzh/temp/ECOS/problem_data_pos_h.npy')

cs = np.vstack((c_negative, c_negpos, c_pos))
hs = np.vstack((h_negative, h_negpos, h_pos))

print('yahoo')
