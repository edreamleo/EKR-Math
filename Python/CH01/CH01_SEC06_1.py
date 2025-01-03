#@+leo-ver=5-thin
#@+node:ekr.20241212100513.35: * @file Python/CH01\CH01_SEC06_1.py
# DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated...
# Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
    # m = int(mat_contents['m'])
    # n = int(mat_contents['n'])

# RuntimeWarning: overflow encountered in scalar add
#   subset = faces[:, sum(nfaces[:person]) : sum(nfaces[: (person + 1)])]
# Traceback (most recent call last):
  # File "c:\Repos\EKR-Math\Python\CH01\CH01_SEC06_1.py", line 41, in <module>
    # allFaces[j * n : (j + 1) * n, k * m : (k + 1) * m] = np.reshape(subset[:, count], (m, n)).T                                                            # ~~~~~~^^^^^^^^^^
# IndexError: index 0 is out of bounds for axis 1 with size 0

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
#@+others
#@+node:ekr.20241222043421.1: ** Data
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 18})

mat_contents = scipy.io.loadmat(os.path.join('..', 'DATA', 'allFaces.mat'))
faces = mat_contents['faces']
m = int(mat_contents['m'])
n = int(mat_contents['n'])
nfaces = np.ndarray.flatten(mat_contents['nfaces'])
allPersons = np.zeros((n * 6, m * 6))
#@+node:ekr.20241212100513.37: ** Faces 1
count = 0
for j in range(6):
    for k in range(6):
        allPersons[j * n : (j + 1) * n, k * m : (k + 1) * m] = np.reshape(faces[:, np.sum(nfaces[:count])], (m, n)).T
        count += 1

img = plt.imshow(allPersons)
img.set_cmap('gray')
plt.axis('off')
plt.show()

#@+node:ekr.20241212100513.38: ** Faces 2
for person in range(len(nfaces)):
    subset = faces[:, sum(nfaces[:person]) : sum(nfaces[: (person + 1)])]

    allFaces = np.zeros((n * 8, m * 8))
    count = 0
    try:
        for j in range(8):
            for k in range(8):
                if count < nfaces[person]:
                    allFaces[j * n : (j + 1) * n, k * m : (k + 1) * m] = np.reshape(subset[:, count], (m, n)).T
                    count += 1
    except IndexError as e:
        print(repr(e))

    img = plt.imshow(allFaces)
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()
#@-others
#@@language python
#@@tabwidth -4
#@-leo
