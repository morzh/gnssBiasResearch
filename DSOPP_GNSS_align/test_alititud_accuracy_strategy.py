import numpy as np
import matplotlib.pyplot as plt


altitude_accuracy = np.load("/home/morzh/work/DSOPP_tests_data_temp/altitude_accuracy/altitude_accuracy.npy")
altitude_threshold = np.load("/home/morzh/work/DSOPP_tests_data_temp/altitude_accuracy/altitude_accuracy_threshold_mask.npy")

abscissa = np.linspace(0, altitude_accuracy.size - 1, altitude_accuracy.size)

plt.plot(abscissa, altitude_accuracy)
plt.plot(abscissa, altitude_threshold)
plt.tight_layout()
plt.show()