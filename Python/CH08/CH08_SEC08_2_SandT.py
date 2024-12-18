#@+leo-ver=5-thin
#@+node:ekr.20241212100516.100: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH08\CH08_SEC08_2_SandT.py
#@+others
#@+node:ekr.20241212100516.102: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from control.matlab import *
import slycot
from scipy import signal
# Python control toolbox available at https://python-control.readthedocs.io/

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})


#@+node:ekr.20241212100516.103: ** s = tf(np.array([1,0]),np.array([0,1]))
s = tf(np.array([1, 0]), np.array([0, 1]))
L = 1 / s
S = (1 / (1 + L))
T = L / (1 + L)
_, _, _ = bode(L, label='L')
_, _, _ = bode(S, label='S')
_, _, _ = bode(T, label='T')
plt.legend()
plt.show()

#@+node:ekr.20241212100516.104: ** Cell 3
#@-others
#@@language python
#@@tabwidth -4
#@-leo
