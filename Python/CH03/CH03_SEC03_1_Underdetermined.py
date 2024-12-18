#@+leo-ver=5-thin
#@+node:ekr.20241212100514.119: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH03\CH03_SEC03_1_Underdetermined.py
#@+others
#@+node:ekr.20241212100514.121: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
plt.rcParams['figure.figsize'] = [12, 18]
plt.rcParams.update({'font.size': 18})

# Solve y = Theta * s for "s"
n = 1000  # dimension of s
p = 200  # number of measurements, dim(y)
Theta = np.random.randn(p, n)
y = np.random.randn(p)

# L1 Minimum norm solution s_L1
def L1_norm(x):
    return np.linalg.norm(x, ord=1)

constr = ({'type': 'eq', 'fun': lambda x:  Theta @x - y})
x0 = np.linalg.pinv(Theta) @y  # initialize with L2 solution
res = minimize(L1_norm, x0, method='SLSQP', constraints=constr)
s_L1 = res.x

#@+node:ekr.20241212100514.122: ** L2 Minimum norm solution s_L2
# L2 Minimum norm solution s_L2
s_L2 = np.linalg.pinv(Theta) @y

#@+node:ekr.20241212100514.123: ** fig,axs = plt.subplots(2,2)
fig, axs = plt.subplots(2, 2)
axs = axs.reshape(-1)
axs[0].plot(s_L1, color='b', LineWidth=1.5)
axs[0].set_ylim(-0.2, 0.2)
axs[1].plot(s_L2, color='r', LineWidth=1.5)
axs[1].set_ylim(-0.2, 0.2)
axs[2].hist(s_L1, bins=np.arange(-0.105, 0.105, 0.01), rwidth=0.9)
axs[3].hist(s_L2, bins=np.arange(-0.105, 0.105, 0.01), rwidth=0.9)

plt.show()

#@+node:ekr.20241212100514.124: ** Cell 4
#@+node:ekr.20241212100514.125: ** Cell 5
#@-others
#@@language python
#@@tabwidth -4
#@-leo
