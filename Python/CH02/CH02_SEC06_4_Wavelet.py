#@+leo-ver=5-thin
#@+node:ekr.20241212100514.101: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH02\CH02_SEC06_4_Wavelet.py
#@+others
#@+node:ekr.20241212100514.103: ** Using the PyWavelets module, available
# Using the PyWavelets module, available at
# https://pywavelets.readthedocs.io/en/latest/install.html

from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
plt.rcParams['figure.figsize'] = [16, 16]
plt.rcParams.update({'font.size': 18})

A = imread(os.path.join('..', 'DATA', 'dog.jpg'))
B = np.mean(A, -1);  # Convert RGB to grayscale

#@+node:ekr.20241212100514.104: ** # Wavelet decomposition (2 level)
## Wavelet decomposition (2 level)
n = 2
w = 'db1'
coeffs = pywt.wavedec2(B, wavelet=w, level=n)

# normalize each coefficient array
coeffs[0] /= np.abs(coeffs[0]).max()
for detail_level in range(n):
    coeffs[detail_level + 1] = [d / np.abs(d).max() for d in coeffs[detail_level + 1]]

arr, coeff_slices = pywt.coeffs_to_array(coeffs)



plt.imshow(arr, cmap='gray', vmin=-0.25, vmax=0.75)
plt.show()
#@-others
#@@language python
#@@tabwidth -4
#@-leo
