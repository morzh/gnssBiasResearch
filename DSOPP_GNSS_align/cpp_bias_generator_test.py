import numpy as np
import matplotlib.pyplot as plt


bias = np.load('/home/morzh/work/DSOPP_tests_data_temp/test_bias_value.npy')
accuracy = np.load('/home/morzh/work/DSOPP_tests_data_temp/test_bias_accuracy.npy')
abscissa = np.linspace(0, bias.shape[0]-1.0, bias.size)


plt.figure(figsize=(20, 10))
plt.plot(abscissa, bias)
plt.plot(abscissa, accuracy)
plt.show()