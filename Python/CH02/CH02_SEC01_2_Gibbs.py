#@+leo-ver=5-thin
#@+node:ekr.20241212100514.37: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH02\CH02_SEC01_2_Gibbs.py
#@+others
#@+node:ekr.20241212100514.39: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

dx = 0.01
L = 2 * np.pi
x = np.arange(0, L + dx, dx)
n = len(x)
nquart = int(np.floor(n / 4))

f = np.zeros_like(x)
f[nquart : 3 * nquart] = 1

A0 = np.sum(f * np.ones_like(x)) * dx * 2 / L
fFS = A0 / 2 * np.ones_like(f)

for k in range(1, 101):
    Ak = np.sum(f * np.cos(2 * np.pi * k * x / L)) * dx * 2 / L
    Bk = np.sum(f * np.sin(2 * np.pi * k * x / L)) * dx * 2 / L
    fFS = fFS + Ak * np.cos(2 * k * np.pi * x / L) + Bk * np.sin(2 * k * np.pi * x / L)

plt.plot(x, f, color='k', LineWidth=2)
plt.plot(x, fFS, '-', color='r', LineWidth=1.5)
plt.show()

#@+node:ekr.20241212100514.40: ** Cell 2
#@+node:ekr.20241212100514.41: ** Cell 3
#@-others
#@@language python
#@@tabwidth -4
#@-leo
