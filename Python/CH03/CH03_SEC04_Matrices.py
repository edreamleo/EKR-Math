#@+leo-ver=5-thin
#@+node:ekr.20241212100514.137: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH03\CH03_SEC04_Matrices.py
#@+others
#@+node:ekr.20241212100514.139: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
import scipy.io
from scipy.fftpack import dct, idct
from scipy.optimize import minimize


plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})

mat = scipy.io.loadmat(os.path.join('..', 'DATA', 'CC2.mat'))
CC = mat['CC']
CC_map = ListedColormap(CC)

p = 14
n = 32

#@+node:ekr.20241212100514.140: ** # Plot Psi
## Plot Psi
# def padflip(X):
#     nx,ny = X.shape
#     X = np.flipud(X)
#     Y = np.zeros((nx+1,ny+1))
#     Y[:-1,:-1] = X
#     return Y

Psi = dct(np.identity(n))
plt.pcolor(np.flipud(Psi), cmap=CC_map)
plt.axis('off')
plt.show()

#@+node:ekr.20241212100514.141: ** # Plot C
## Plot C
plt.rcParams['figure.figsize'] = [12, 6]
fig, ax = plt.subplots(1, 1)
C = np.identity(n)
perm = np.random.permutation(n)[:p]
C = C[perm, :]  # compressed measurement
plt.pcolor(np.flipud(C), cmap=CC_map)
plt.grid(True)
plt.xticks(np.arange(n))
plt.yticks(np.arange(len(perm)))
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()

#@+node:ekr.20241212100514.142: ** # Plot Theta
## Plot Theta

Theta = C @Psi
plt.pcolor(np.flipud(Theta), cmap=CC_map)
plt.axis('off')
plt.show()

#@+node:ekr.20241212100514.143: ** # Plot s, y
## Plot s, y
s = np.zeros((n, 1))
s[2] = 1.4
s[13] = 0.7
s[27] = 2.2
y = C @Psi @s

fig, axs = plt.subplots(1, 4)
axs[0].pcolor(np.flipud(s), cmap=CC_map)
axs[0].set_xlabel('s')

sL2 = np.linalg.pinv(Theta) @y
axs[1].pcolor(np.flipud(sL2), cmap=CC_map)
axs[1].set_xlabel('sL2')

sbackslash = np.linalg.lstsq(Theta, y)[0]
axs[2].pcolor(np.flipud(sbackslash), cmap=CC_map)
axs[2].set_xlabel('sbackslash')

axs[3].pcolor(np.flipud(y), cmap=CC_map)
axs[3].set_xlabel('y')

for ax in axs:
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.show()


#@+node:ekr.20241212100514.144: ** # L1-Minimization using SciPy
## L1-Minimization using SciPy
def L1_norm(x):
    return np.linalg.norm(x, ord=1)

y = y.reshape(-1)
constr = ({'type': 'eq', 'fun': lambda x:  Theta @x - y})
x0 = np.linalg.pinv(Theta) @y
res = minimize(L1_norm, x0, method='SLSQP', constraints=constr)
s2 = res.x

#@+node:ekr.20241212100514.145: ** # Plot C and Theta (2) - Gaussian Random
## Plot C and Theta (2) - Gaussian Random
plt.rcParams['figure.figsize'] = [8, 4]

C = np.random.randn(p, n)

plt.figure()
plt.pcolor(np.flipud(C), cmap=CC_map)
plt.axis('off')
plt.show()

Theta = C @Psi
plt.figure()
plt.pcolor(np.flipud(Theta), cmap=CC_map)
plt.axis('off')
plt.show()

#@+node:ekr.20241212100514.146: ** Plot C and Theta (3) - Bernoulli Random
# Plot C and Theta (3) - Bernoulli Random
C = np.random.randn(p, n)
C = C > 0

plt.figure()
plt.pcolor(np.flipud(C), cmap=CC_map)
plt.axis('off')
plt.show()

plt.figure()
Theta = C @Psi
plt.pcolor(np.flipud(Theta), cmap=CC_map)
plt.axis('off')
plt.show()

#@+node:ekr.20241212100514.147: ** Plot C and Theta (4) - Sparse Bernoulli
# Plot C and Theta (4) - Sparse Bernoulli
C = np.random.randn(p, n)
C = C > 1

plt.figure()
plt.pcolor(np.flipud(C), cmap=CC_map)
plt.axis('off')
plt.show()

plt.figure()
Theta = C @Psi
plt.pcolor(np.flipud(Theta), cmap=CC_map)
plt.axis('off')
plt.show()

#@+node:ekr.20241212100514.148: ** Bad C and Theta (5) - DCT Meas
# Bad C and Theta (5) - DCT Meas
C = idct(np.identity(n))
perm = np.arange(n - p, n)
C = C[perm, :]  # compressed measurement

plt.figure()
plt.pcolor(np.flipud(C), cmap=CC_map)
plt.axis('off')
plt.show()

plt.figure()
Theta = C @Psi
plt.pcolor(np.flipud(Theta), cmap=CC_map)
plt.axis('off')
plt.show()

plt.figure()
y = Theta @s
plt.pcolor(np.flipud(y), cmap=CC_map)
plt.axis('off')
plt.show()

#@+node:ekr.20241212100514.149: ** Cell 11
#@+node:ekr.20241212100514.150: ** Cell 12
#@+node:ekr.20241212100514.151: ** Cell 13
#@-others
#@@language python
#@@tabwidth -4
#@-leo
