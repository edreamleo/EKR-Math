#@+leo-ver=5-thin
#@+node:ekr.20241212100514.8: * @file Python/CH01\CH01_SEC07_3.py
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import skimage.transform  ### pip install skikit-image

#@+others
#@+node:ekr.20241212100514.10: ** Plot
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams.update({'font.size': 18})

n = 1000
q = int(n / 4)
X = np.zeros((n, n))
X[(q - 1) : (3 * q), (q - 1) : (3 * q)] = 1

nAngles = 12  # Sweep through 12 different angles, from 0:4:44 degrees
cm_np = np.array([[0, 0, 2 / 3],
                 [0, 0, 1],
                 [0, 1 / 3, 1],
                 [0, 2 / 3, 1],
                 [0, 1, 1],
                 [1 / 3, 1, 2 / 3],
                 [2 / 3, 1, 1 / 3],
                 [1, 1, 0],
                 [1, 2 / 3, 0],
                 [1, 1 / 3, 0],
                 [1, 0, 0],
                 [2 / 3, 0, 0]])


cmap = plt.cm.jet
cmap.set_bad(alpha=0.0)

U, S, VT = np.linalg.svd(X, full_matrices=0)

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
plt.imshow(np.ma.masked_where(X == 0, X), vmin=0, vmax=nAngles)
plt.set_cmap(cmap)
plt.axis('off')


ax2 = fig1.add_subplot(122)
plt.semilogy(S, '-o', color=tuple(cm_np[0]))
plt.grid()

plt.show()

#@+node:ekr.20241212100514.11: ** Xrot = X
Xrot = X

fig, axs = plt.subplots(1, 2)

for j in range(nAngles):
    Xrot = skimage.transform.rotate(X, j * 4)  #rotate by theta = j*4 degrees
    Xrot[np.nonzero(Xrot)] = j

    U, S, VT = np.linalg.svd(Xrot)

    axs[0].imshow(np.ma.masked_where(Xrot == 0, Xrot), vmin=0, vmax=nAngles)
    plt.set_cmap(cmap)
    axs[0].axis('off')

    axs[1].semilogy(S, '-o', color=tuple(cm_np[j]))
    axs[1].axis('on')
    axs[1].grid(1)

plt.show()

#@+node:ekr.20241212100514.12: ** Cell 3
#@-others
#@@language python
#@@tabwidth -4
#@-leo
