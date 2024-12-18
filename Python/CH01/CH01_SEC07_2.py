#@+leo-ver=5-thin
#@+node:ekr.20241212100514.1: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH01\CH01_SEC07_2.py
#@+others
#@+node:ekr.20241212100514.3: ** import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

n = 1000
q = int(n / 4)
X = np.zeros((n, n))
X[(q - 1) : (3 * q), (q - 1) : (3 * q)] = 1

plt.imshow(X)
plt.set_cmap('gray')
plt.axis('off')
plt.title('Unrotated Matrix')
plt.show()


#@+node:ekr.20241212100514.4: ** X_rot = skimage.transform.rotate(X,10)
X_rot = skimage.transform.rotate(X, 10)
X_rot[np.nonzero(X_rot)] = 1


plt.imshow(Y)
plt.set_cmap('gray')
plt.axis('off')
plt.title('Rotated Matrix')
plt.show()

#@+node:ekr.20241212100514.5: ** U, S, VT =
U, S, VT = np.linalg.svd(X, full_matrices=0)


plt.semilogy(S, '-o', color='k')
plt.ylim((10 ** (-16), 10 ** (4) + 1))
plt.yticks(np.power(10, np.arange(-16, 4, 4, dtype=float)))
plt.xticks(np.arange(0, 1000, 250))
plt.grid()
plt.title('Unrotated Matrix: Spectrum')
plt.show()

#@+node:ekr.20241212100514.6: ** U_rot, S_rot, VT_rot =
U_rot, S_rot, VT_rot = np.linalg.svd(X_rot, full_matrices=0)

plt.semilogy(S_rot, '-o', color='k')
plt.ylim((10 ** (-16), 10 ** (4) + 1))
plt.yticks(np.power(10, np.arange(-16, 4, 4, dtype=float)))
plt.xticks(np.arange(0, 1000, 250))
plt.grid()
plt.title('Rotated Matrix: Spectrum')
plt.show()

#@+node:ekr.20241212100514.7: ** Cell 5
#@-others
#@@language python
#@@tabwidth -4
#@-leo
