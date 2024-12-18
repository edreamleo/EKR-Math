#@+leo-ver=5-thin
#@+node:ekr.20241212100516.94: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH08\CH08_SEC08_1_TransferFunction.py
#@+others
#@+node:ekr.20241212100516.96: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from control.matlab import *
import slycot
from scipy import signal
# Python control toolbox available at https://python-control.readthedocs.io/

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})


#@+node:ekr.20241212100516.97: ** s = tf(np.array([1,0]),np.array([0,1]))
s = tf(np.array([1, 0]), np.array([0, 1]))
G = 1 / (s ** 2 + s + 2)
w, mag, phase = bode(G)

#@+node:ekr.20241212100516.98: ** A = np.array([[0,1],[-2,-1]])
A = np.array([[0, 1], [-2, -1]])
B = np.array([0, 1]).reshape((2, 1))
C = np.array([1, 0])
D = 0
G = ss2tf(A, B, C, D)

ia, it = impulse(G)

plt.plot(it[[0, -1]], np.array([0, 0]), 'k:')
plt.plot(it, ia)
plt.title('Impulse Response')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.show()

#@+node:ekr.20241212100516.99: ** Cell 4
#@-others
#@@language python
#@@tabwidth -4
#@-leo
