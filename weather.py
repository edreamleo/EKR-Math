#@+leo-ver=5-thin
#@+node:ekr.20241212071918.1: * @file ../../weather.py
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter("ignore")
A = np.array([[0.5, 0.5, 0.25], [0.25, 0.0, 0.25], [0.25, 0.5, 0.5]])
days = 15
weather = np.zeros((3, days))
x = np.array([[1.0], [0.0], [0.0]])
weather[:, 0] = x[:, 0]
for k in range(days):
    x = A @x
    weather[:, k] = x[:, 0]
    if 8 < k < 14:
        print(k)
        print(x)
plt.plot(weather.transpose())
plt.grid(True)
plt.show()
#@-leo
