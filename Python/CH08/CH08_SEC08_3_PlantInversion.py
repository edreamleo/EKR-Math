#@+leo-ver=5-thin
#@+node:ekr.20241212100516.105: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH08\CH08_SEC08_3_PlantInversion.py
#@+others
#@+node:ekr.20241212100516.107: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from control.matlab import *
import slycot
from scipy import signal
# Python control toolbox available at https://python-control.readthedocs.io/

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})


#@+node:ekr.20241212100516.108: ** s = tf(np.array([1,0]),np.array([0,1]))
s = tf(np.array([1, 0]), np.array([0, 1]))
G = (s + 1) / (s - 2)
Gtrue = (s + 0.9) / (s - 1.9)

K = 1 / G

L = K * Gtrue


fig = plt.figure()
gm, pm, wg, wp = margin(L)
_, _, _ = bode(L)
for ax in fig.axes:
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ax.plot(wg * np.ones(2), ax.get_ylim(), 'k--', linewidth=2)
    ax.plot(wp * np.ones(2), ax.get_ylim(), 'k--', linewidth=2)
    ax.plot(ax.get_xlim(), np.zeros(2), 'k--', linewidth=2)
    ax.set_xlim(xl)
    ax.set_ylim(yl)

CL = feedback(L, 1)
CL

#@+node:ekr.20241212100516.109: ** Cell 3
#@+node:ekr.20241212100516.110: ** Cell 4
#@-others
#@@language python
#@@tabwidth -4
#@-leo
