#@+leo-ver=5-thin
#@+node:ekr.20241212100513.31: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH01\CH01_SEC05_2_OvarianCancer.py
#@+others
#@+node:ekr.20241212100513.33: ** import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams.update({'font.size': 18})


obs = np.loadtxt(os.path.join('..', 'DATA', 'ovariancancer_obs.csv'), delimiter=',')

f = open(os.path.join('..', 'DATA', 'ovariancancer_grp.csv'), "r")
grp = f.read().split("\n")

U, S, VT = np.linalg.svd(obs, full_matrices=0)

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax1.semilogy(S, '-o', color='k')
ax2 = fig1.add_subplot(122)
ax2.plot(np.cumsum(S) / np.sum(S), '-o', color='k')

plt.show()

#@+node:ekr.20241212100513.34: ** fig2 = plt.figure()
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')

for j in range(obs.shape[0]):
    x = VT[0, :] @obs[j, :].T
    y = VT[1, :] @obs[j, :].T
    z = VT[2, :] @obs[j, :].T

    if grp[j] == 'Cancer':
        ax.scatter(x, y, z, marker='x', color='r', s=50)
    else:
        ax.scatter(x, y, z, marker='o', color='b', s=50)

ax.view_init(25, 20)
plt.show()
#@-others
#@@language python
#@@tabwidth -4
#@-leo
