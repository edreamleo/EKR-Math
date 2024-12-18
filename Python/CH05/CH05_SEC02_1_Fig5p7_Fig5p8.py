#@+leo-ver=5-thin
#@+node:ekr.20241212100515.79: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH05\CH05_SEC02_1_Fig5p7_Fig5p8.py
#@+others
#@+node:ekr.20241212100515.81: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

#@+node:ekr.20241212100515.82: ** Training and test set sizes
# Training and test set sizes
n1 = 100  # Training set size
n2 = 50  # Test set size

# Random ellipse 1 centered at (-2,0)
x = np.random.randn(n1 + n2) - 2
y = 0.5 * np.random.randn(n1 + n2)

# Random ellipse 5 centered at (1,0)
x5 = 2 * np.random.randn(n1 + n2) + 1
y5 = 0.5 * np.random.randn(n1 + n2)

# Random ellipse 2 centered at (2,-2)
x2 = np.random.randn(n1 + n2) + 2
y2 = 0.2 * np.random.randn(n1 + n2) - 2

# Rotate ellipse 2 by theta
theta = np.pi / 4
A = np.zeros((2, 2))
A[0, 0] = np.cos(theta)
A[0, 1] = -np.sin(theta)
A[1, 0] = np.sin(theta)
A[1, 1] = np.cos(theta)

x3 = A[0, 0] * x2 + A[0, 1] * y2
y3 = A[1, 0] * x2 + A[1, 1] * y2

#@+node:ekr.20241212100515.83: ** fig,axs = plt.subplots(2,2)
fig, axs = plt.subplots(2, 2)
axs = axs.reshape(-1)

axs[0].plot(x[:n1], y[:n1], 'o', markerfacecolor=(0.8, 0.8, 0.8), markeredgecolor='k', ms=15)
axs[0].plot(x3[:n1], y3[:n1], 'o', markerfacecolor=(0.8, 0.8, 0.8), markeredgecolor='k', ms=15)


axs[1].plot(x[:70], y[:70], 'o', markerfacecolor=(0.8, 0.8, 0.8), markeredgecolor='k', ms=15)
axs[1].plot(x3[:70], y3[:70], 'o', markerfacecolor=(0.8, 0.8, 0.8), markeredgecolor='k', ms=15)
axs[1].plot(x[70:100], y[70:100], 'o', markerfacecolor=(0, 1, 0.2), markeredgecolor='k', ms=15)
axs[1].plot(x3[70:100], y3[70:100], 'o', markerfacecolor=(0.9, 0, 1), markeredgecolor='k', ms=15)

axs[2].plot(x5[:n1], y5[:n1], 'o', markerfacecolor=(0.8, 0.8, 0.8), markeredgecolor='k', ms=15)
axs[2].plot(x3[:n1], y3[:n1], 'o', markerfacecolor=(0.8, 0.8, 0.8), markeredgecolor='k', ms=15)

axs[3].plot(x5[:70], y5[:70], 'o', markerfacecolor=(0.8, 0.8, 0.8), markeredgecolor='k', ms=15)
axs[3].plot(x3[:70], y3[:70], 'o', markerfacecolor=(0.8, 0.8, 0.8), markeredgecolor='k', ms=15)
axs[3].plot(x5[70:100], y5[70:100], 'o', markerfacecolor=(0, 1, 0.2), markeredgecolor='k', ms=15)
axs[3].plot(x3[70:100], y3[70:100], 'o', markerfacecolor=(0.9, 0, 1), markeredgecolor='k', ms=15)

for ax in axs:
    ax.set_xlim(-6, 6)
    ax.set_ylim(-2, 2)

plt.show()

#@+node:ekr.20241212100515.84: ** training set size
n1 = 300  # training set size
x1 = 1.5 * np.random.randn(n1) - 1.5
y1 = 1.2 * np.random.randn(n1) + np.power(x1 + 1.5, 2) - 7
x2 = 1.5 * np.random.randn(n1) + 1.5
y2 = 1.2 * np.random.randn(n1) - np.power(x2 - 1.5, 2) + 7

fig, axs = plt.subplots(2)
axs[0].plot(x1, y1, 'o', markerfacecolor=(0, 1, 0.2), markeredgecolor='k', ms=15)
axs[0].plot(x2, y2, 'o', markerfacecolor=(0.9, 0, 1), markeredgecolor='k', ms=15)
axs[0].set_xlim(-6, 6)
axs[0].set_ylim(-12, 12)

r = 7 + np.random.randn(n1)
th = 2 * np.pi * np.random.randn(n1)
xr = r * np.cos(th)
yr = r * np.sin(th)

x5 = np.random.randn(n1)
y5 = np.random.randn(n1)

axs[1].plot(xr, yr, 'o', markerfacecolor=(0, 1, 0.2), markeredgecolor='k', ms=15)
axs[1].plot(x5, y5, 'o', markerfacecolor=(0.9, 0, 1), markeredgecolor='k', ms=15)
axs[1].set_xlim(-10, 10)
axs[1].set_ylim(-10, 10)

plt.show()

#@+node:ekr.20241212100515.85: ** Cell 5
#@-others
#@@language python
#@@tabwidth -4
#@-leo
