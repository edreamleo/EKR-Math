#@+leo-ver=5-thin
#@+node:ekr.20241212100514.21: * @file Python/CH01\CH01_SEC09_Tensor.py
import matplotlib.pyplot as plt
import numpy as np
#### import os
from matplotlib import animation  ###, rc
from IPython.display import HTML
# Tensor factorization method requires the TensorLy module,
# available at http://tensorly.org/stable/installation.html
from tensorly.decomposition import parafac

#@+others
#@+node:ekr.20241212100514.23: ** animation
# # %matplotlib inline

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})
plt.rcParams['animation.html'] = 'jshtml'

x = np.arange(-5, 5.01, 0.1)
y = np.arange(-6, 6.01, 0.1)
t = np.arange(0, 10 * np.pi + 0.1, 0.1)

X, Y, T = np.meshgrid(x, y, t)

A = np.exp(-(X ** 2 + 0.5 * Y ** 2)) * np.cos(2 * T) + \
    (np.divide(np.ones_like(X), np.cosh(X)) * np.tanh(X) * np.exp(-0.2 * Y ** 2)) * np.sin(T)

fig = plt.figure()

# ValueError: For X (102) and Y (122) with flat shading,
# A should have shape (121, 101, 3) or (121, 101, 4) or (121, 101) or (12221,), not (0,)

pcm = plt.pcolormesh(X[:,:, 0], Y[:,:, 0], A[:,:, 0], vmin=-1, vmax=1)
    # , shading='interp')

def init():
    pcm.set_array(np.array([]))
    return pcm

def animate(iter):
    pcm.set_array(A[: -1, : -1, iter].ravel())
    return pcm

anim = animation.FuncAnimation(fig, animate,
    init_func=init,
    frames=len(t),
    interval=50, blit=False,
    repeat=False)

HTML(anim.to_jshtml())
#@+node:ekr.20241212100514.24: ** plot 1
plt.rcParams['figure.figsize'] = [16, 10]

fig, axs = plt.subplots(2, 4)
axs = axs.reshape(-1)

for j in range(8):
    plt.sca(axs[j])
    plt.pcolormesh(X[:,:, 0], Y[:,:, 0], A[:, :, 8 * (j + 1) - 3], vmin=-1, vmax=1, shading='interp')
    axs[j].axis('off')
    plt.set_cmap('hot')

#@+node:ekr.20241212100514.25: ** plot 2

plt.rcParams['figure.figsize'] = [12, 12]

A1, A2, A3 = parafac(A, 2)

fig, axs = plt.subplots(3, 1)
axs[0].plot(y, A1, linewidth=2)
axs[1].plot(x, A2, linewidth=2)
axs[2].plot(t, A3, linewidth=2)
plt.show()

#@-others
#@@language python
#@@tabwidth -4
#@-leo
