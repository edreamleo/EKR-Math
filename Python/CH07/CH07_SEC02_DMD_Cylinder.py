#@+leo-ver=5-thin
#@+node:ekr.20241212100516.22: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH07\CH07_SEC02_DMD_Cylinder.py
#@+others
#@+node:ekr.20241212100516.24: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import io
import os

rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [8, 16]

#@+node:ekr.20241212100516.25: ** vortall_mat = io.loadmat(os.path.join('.
vortall_mat = io.loadmat(os.path.join('..', 'DATA', 'VORTALL.mat'))
X = vortall_mat['VORTALL']
# VORTALL contains flow fields reshaped into column vectors

#@+node:ekr.20241212100516.26: ** def DMD(X,Xprime,r):
def DMD(X, Xprime, r):
    U, Sigma, VT = np.linalg.svd(X, full_matrices=0)  # Step 1
    Ur = U[:,:r]
    Sigmar = np.diag(Sigma[:r])
    VTr = VT[: r, :]
    Atilde = np.linalg.solve(Sigmar.T, (Ur.T @Xprime @VTr.T).T).T  # Step 2
    Lambda, W = np.linalg.eig(Atilde)  # Step 3
    Lambda = np.diag(Lambda)

    Phi = Xprime @np.linalg.solve(Sigmar.T, VTr).T @W  # Step 4
    alpha1 = Sigmar @VTr[:, 0]
    b = np.linalg.solve(W @Lambda, alpha1)
    return Phi, Lambda, b


#@+node:ekr.20241212100516.27: ** Phi, Lambda, b =
Phi, Lambda, b = DMD(X[:,:-1], X[:, 1 :], 21)

#@+node:ekr.20241212100516.28: ** # Plot Mode 2
## Plot Mode 2
vortmin = -5
vortmax = 5
V2 = np.copy(np.real(np.reshape(Phi[:, 1], (449, 199))))
V2 = V2.T

# normalize values... not symmetric
minval = np.min(V2)
maxval = np.max(V2)

if np.abs(minval) < 5 and np.abs(maxval) < 5:
    if np.abs(minval) > np.abs(maxval):
        vortmax = maxval
        vortmin = -maxval
    else:
        vortmin = minval
        vortmax = -minval

V2[V2 > vortmax] = vortmax
V2[V2 < vortmin] = vortmin

plt.imshow(V2, cmap='jet', vmin=vortmin, vmax=vortmax)

cvals = np.array([-4, -2, -1, -0.5, -0.25, -0.155])
plt.contour(V2, cvals * vortmax / 5, colors='k', linestyles='dashed', linewidths=1)
plt.contour(V2, np.flip(-cvals) * vortmax / 5, colors='k', linestyles='solid', linewidths=0.4)

plt.scatter(49, 99, 5000, color='k')  # draw cylinder


plt.show()

#@+node:ekr.20241212100516.29: ** V2 =
V2 = np.real(np.reshape(Phi[:, 1], (199, 449)))

# plt.hist(np.real(Phi).reshape(-1),128)
plt.hist(V2.reshape(-1), 128)
plt.show()

#@+node:ekr.20241212100516.30: ** Cell 7
#@+node:ekr.20241212100516.31: ** Cell 8
#@-others
#@@language python
#@@tabwidth -4
#@-leo
