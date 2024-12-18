#@+leo-ver=5-thin
#@+node:ekr.20241212100514.185: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH03\CH03_SEC07_RPCA.py
#@+others
#@+node:ekr.20241212100514.187: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io

plt.rcParams['figure.figsize'] = [7, 7]
plt.rcParams.update({'font.size': 18})

mat = scipy.io.loadmat(os.path.join('..', 'DATA', 'allFaces.mat'))
faces = mat['faces']
nfaces = mat['nfaces'].reshape(-1)


#@+node:ekr.20241212100514.188: ** # Function Definitions
## Function Definitions

def shrink(X, tau):
    Y = np.abs(X) - tau
    return np.sign(X) * np.maximum(Y, np.zeros_like(Y))
def SVT(X, tau):
    U, S, VT = np.linalg.svd(X, full_matrices=0)
    out = U @np.diag(shrink(S, tau)) @VT
    return out
def RPCA(X):
    n1, n2 = X.shape
    mu = n1 * n2 / (4 * np.sum(np.abs(X.reshape(-1))))
    lambd = 1 / np.sqrt(np.maximum(n1, n2))
    thresh = 10 ** (-7) * np.linalg.norm(X)

    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    L = np.zeros_like(X)
    count = 0
    while (np.linalg.norm(X - L - S) > thresh) and (count < 1000):
        L = SVT(X - S + (1 / mu) * Y, 1 / mu)
        S = shrink(X - L + (1 / mu) * Y, lambd / mu)
        Y = Y + mu * (X - L - S)
        count += 1
    return L, S


#@+node:ekr.20241212100514.189: ** X = faces[:,:nfaces[0]]
X = faces[:, : nfaces[0]]
L, S = RPCA(X)

#@+node:ekr.20241212100514.190: ** inds = (3,4,14,15,17,18,19,20,21,32,43)
inds = (3, 4, 14, 15, 17, 18, 19, 20, 21, 32, 43)

for k in inds:
    fig, axs = plt.subplots(2, 2)
    axs = axs.reshape(-1)
    axs[0].imshow(np.reshape(X[:, k - 1], (168, 192)).T, cmap='gray')
    axs[0].set_title('X')
    axs[1].imshow(np.reshape(L[:, k - 1], (168, 192)).T, cmap='gray')
    axs[1].set_title('L')
    axs[2].imshow(np.reshape(S[:, k - 1], (168, 192)).T, cmap='gray')
    axs[2].set_title('S')
    for ax in axs:
        ax.axis('off')

#@+node:ekr.20241212100514.191: ** Cell 5
#@+node:ekr.20241212100514.192: ** Cell 6
#@-others
#@@language python
#@@tabwidth -4
#@-leo
