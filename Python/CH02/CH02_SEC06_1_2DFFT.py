#@+leo-ver=5-thin
#@+node:ekr.20241212100514.88: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH02\CH02_SEC06_1_2DFFT.py
#@+others
#@+node:ekr.20241212100514.90: ** from matplotlib.image import imread
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 18})

A = imread(os.path.join('..', 'DATA', 'dog.jpg'))
B = np.mean(A, -1);  # Convert RGB to grayscale

fig, axs = plt.subplots(1, 3)

# Plot image
img = axs[0].imshow(B)
img.set_cmap('gray')
axs[0].axis('off')

# Compute row-wise FFT

Cshift = np.zeros_like(B, dtype='complex_')
C = np.zeros_like(B, dtype='complex_')

for j in range(B.shape[0]):
    Cshift[j, :] = np.fft.fftshift(np.fft.fft(B[j, :]))
    C[j, :] = np.fft.fft(B[j, :])

img = axs[1].imshow(np.log(np.abs(Cshift)))
img.set_cmap('gray')
axs[1].axis('off')

# Compute column-wise FFT

D = np.zeros_like(C)
for j in range(C.shape[1]):
    D[:, j] = np.fft.fft(C[:, j])

img = axs[2].imshow(np.fft.fftshift(np.log(np.abs(D))))
img.set_cmap('gray')
axs[2].axis('off')

plt.show()

# Much more efficient to use fft2
D = np.fft.fft2(B)

#@+node:ekr.20241212100514.91: ** Cell 2
#@-others
#@@language python
#@@tabwidth -4
#@-leo
