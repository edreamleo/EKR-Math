#@+leo-ver=5-thin
#@+node:ekr.20241212100514.92: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH02\CH02_SEC06_2_Compress.py
#@+others
#@+node:ekr.20241212100514.94: ** from matplotlib.image import imread
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 18})

A = imread(os.path.join('..', 'DATA', 'dog.jpg'))
B = np.mean(A, -1);  # Convert RGB to grayscale

Bt = np.fft.fft2(B)
Btsort = np.sort(np.abs(Bt.reshape(-1)))  # sort by magnitude

# Zero out all small coefficients and inverse transform
for keep in (0.1, 0.05, 0.01, 0.002):
    thresh = Btsort[int(np.floor((1 - keep) * len(Btsort)))]
    ind = np.abs(Bt) > thresh  # Find small indices
    Atlow = Bt * ind  # Threshold small indices
    Alow = np.fft.ifft2(Atlow).real  # Compressed image
    plt.figure()
    plt.imshow(Alow, cmap='gray')
    plt.axis('off')
    plt.title('Compressed image: keep = ' + str(keep))

#@+node:ekr.20241212100514.95: ** Cell 2
#@+node:ekr.20241212100514.96: ** Cell 3
#@-others
#@@language python
#@@tabwidth -4
#@-leo
