#@+leo-ver=5-thin
#@+node:ekr.20241212100514.97: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH02\CH02_SEC06_3_Denoise.py
#@+others
#@+node:ekr.20241212100514.99: ** from matplotlib.image import imread
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.figsize'] = [16, 16]
plt.rcParams.update({'font.size': 18})

A = imread(os.path.join('..', 'DATA', 'dog.jpg'))
B = np.mean(A, -1);  # Convert RGB to grayscale

#@+node:ekr.20241212100514.100: ** # Denoise
## Denoise
Bnoise = B + 200 * np.random.randn(*B.shape).astype('uint8')  # Add some noise
Bt = np.fft.fft2(Bnoise)
Btshift = np.fft.fftshift(Bt)
F = np.log(np.abs(Btshift) + 1)  # Put FFT on log scale

fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(Bnoise, cmap='gray')
axs[0, 0].axis('off')

axs[0, 1].imshow(F, cmap='gray')
axs[0, 1].axis('off')

nx, ny = B.shape
X, Y = np.meshgrid(np.arange(-ny / 2 + 1, ny / 2 + 1), np.arange(-nx / 2 + 1, nx / 2 + 1))
# xgrid = np.fft.ifftshift(np.arange(-nx/2+1,nx/2+1))
# ygrid = np.fft.ifftshift(np.arange(-ny/2+1,ny/2+1))
# X,Y = np.meshgrid(ygrid,xgrid)
R2 = np.power(X, 2) + np.power(Y, 2)
ind = R2 < 150 ** 2
Btshiftfilt = Btshift * ind
Ffilt = np.log(np.abs(Btshiftfilt) + 1)  # Put FFT on log scale

axs[1, 1].imshow(Ffilt, cmap='gray')
axs[1, 1].axis('off')

Btfilt = np.fft.ifftshift(Btshiftfilt)
Bfilt = np.fft.ifft2(Btfilt).real
axs[1, 0].imshow(Bfilt, cmap='gray')
axs[1, 0].axis('off')

plt.show()
#@-others
#@@language python
#@@tabwidth -4
#@-leo
