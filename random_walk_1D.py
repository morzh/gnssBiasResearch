import numpy as np
import matplotlib.pyplot as plt


def Randwalk(n):
    x = 0
    y = 0

    time = [x]
    position = [y]
    y=0

    for i in range(1, n + 1):
        move = np.random.normal(0, 1)
        y += move
        time.append(i)
        position.append(y)

    return [time, position]



Randwalk1 = Randwalk(1000)
Randwalk2 = Randwalk(1000)
Randwalk3 = Randwalk(1000)
plt.plot(Randwalk1[0],Randwalk1[1],'r-', label = "Randwalk1")
plt.plot(Randwalk2[0],Randwalk2[1],'g-', label = "Randwalk2")
plt.plot(Randwalk3[0],Randwalk3[1],'b-', label = "Randwalk3")
plt.title("1-D Random Walks")
plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), fancybox=True, shadow=True)
plt.show()