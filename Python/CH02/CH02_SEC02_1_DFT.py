#@+leo-ver=5-thin
#@+node:ekr.20241212100514.47: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH02\CH02_SEC02_1_DFT.py
#@+others
#@+node:ekr.20241212100514.49: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

n = 256
w = np.exp(-1j * 2 * np.pi / n)

#@+node:ekr.20241212100514.50: ** DFT = np.zeros((n,n))
DFT = np.zeros((n, n))

# Slow
for i in range(n):
    for k in range(n):
        DFT[i, k] = w ** (i * k)

DFT = np.real(DFT)

plt.imshow(DFT)
plt.show()

#@+node:ekr.20241212100514.51: ** Fast
# Fast
J, K = np.meshgrid(np.arange(n), np.arange(n))
DFT = np.power(w, J * K)
DFT = np.real(DFT)

plt.imshow(DFT)
plt.show()

#@+node:ekr.20241212100514.52: ** Cell 4
#@-others
#@@language python
#@@tabwidth -4
#@-leo
