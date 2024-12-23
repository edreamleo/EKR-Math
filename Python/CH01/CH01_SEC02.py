#@+leo-ver=5-thin
#@+node:ekr.20241212122548.3: * @file Python/CH01/CH01_SEC02.py
import os
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

figure_number = 1

#@+others
#@+node:ekr.20241212125134.1: ** read the dog, compute values
plt.rcParams['figure.figsize'] = [16, 8]
A = imread(os.path.join('..', 'DATA', 'dog.jpg'))
X = np.mean(A, -1)  # Convert RGB to grayscale
U, S, VT = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)
#@+node:ekr.20241212122548.6: ** plot dog
img = plt.imshow(X)
img.set_cmap('gray')
plt.figure(figure_number)
figure_number += 1
plt.axis('off')
plt.title('Fido')
plt.show()
#@+node:ekr.20241212122548.7: ** pot approx images
j = 0
for r in (5, 20, 100):
    # Construct approximate image
    Xapprox = U[:,:r] @S[0 : r, : r] @VT[: r, :]

    # plt.figure(j + 1)
    plt.figure(figure_number)
    figure_number += 1

    j += 1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title(f"r = {r!s}")
    plt.show()
#@+node:ekr.20241212122548.8: ** plot singular values
plt.figure(figure_number)
figure_number += 1

plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

plt.figure(figure_number)
figure_number += 1
plt.plot(np.cumsum(np.diag(S)) / np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()
#@-others
# print('Done!', __file__)

#@@language python
#@@tabwidth -4
#@-leo
