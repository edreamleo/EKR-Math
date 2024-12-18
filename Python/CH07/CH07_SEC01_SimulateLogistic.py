#@+leo-ver=5-thin
#@+node:ekr.20241212100516.9: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH07\CH07_SEC01_SimulateLogistic.py
#@+others
#@+node:ekr.20241212100516.11: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

#@+node:ekr.20241212100516.12: ** startval = 1
startval = 1
endval = 4
xvals = np.array([[], []])
n_iter = 1000
n_plot = 100

def logistic(xk, r):
    return r * xk * (1 - xk)

for r in np.arange(startval, endval, 0.00025):
    x = 0.5
    for i in range(n_iter):
        x = logistic(x, r)
        if i == n_iter - n_plot:
            xss = x
        if i > n_iter - n_plot:
            xvals = np.append(xvals, np.array([[r], [x]]), axis=1)
            if np.abs(x - xss) < 0.001:
                break


#@+node:ekr.20241212100516.13: ** plt.plot(xvals[1,:],xvals[0,:],'.',ms=0.
plt.plot(xvals[1, :], xvals[0, :], '.', ms=0.1, color='k')
plt.xlim(0, 1)
plt.ylim(1, endval)
plt.gca().invert_yaxis()

#@+node:ekr.20241212100516.14: ** plt.plot(xvals[1,:],xvals[0,:],'.',ms=0.
plt.plot(xvals[1, :], xvals[0, :], '.', ms=0.1, color='k')
plt.xlim(0, 1)
plt.ylim(3.45, 4)
plt.gca().invert_yaxis()

#@+node:ekr.20241212100516.15: ** Cell 5
#@-others
#@@language python
#@@tabwidth -4
#@-leo
