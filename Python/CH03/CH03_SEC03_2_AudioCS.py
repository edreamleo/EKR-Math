#@+leo-ver=5-thin
#@+node:ekr.20241212100514.126: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH03\CH03_SEC03_2_AudioCS.py
#@+others
#@+node:ekr.20241212100514.128: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.fftpack import dct, idct
from scipy.optimize import minimize
sys.path.append(os.path.join('..', 'UTILS'))
from cosamp_fn import cosamp
# cosamp function is available at https://github.com/avirmaux/CoSaMP

plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})

#@+node:ekr.20241212100514.129: ** # Generate signal, DCT of signal
## Generate signal, DCT of signal

n = 4096  # points in high resolution signal
t = np.linspace(0, 1, n)
x = np.cos(2 * 97 * np.pi * t) + np.cos(2 * 777 * np.pi * t)
xt = np.fft.fft(x)  # Fourier transformed signal
PSD = xt * np.conj(xt) / n  # Power spectral density

#@+node:ekr.20241212100514.130: ** # Randomly sample signal
## Randomly sample signal
p = 128  # num. random samples, p = n/32
perm = np.floor(np.random.rand(p) * n).astype(int)
y = x[perm]

#@+node:ekr.20241212100514.131: ** # Solve compressed sensing problem
## Solve compressed sensing problem
Psi = dct(np.identity(n))  # Build Psi
Theta = Psi[perm, :]  # Measure rows of Psi

s = cosamp(Theta, y, 10, epsilon=1.e-10, max_iter=10)  # CS via matching pursuit
xrecon = idct(s)  # reconstruct full signal

#@+node:ekr.20241212100514.132: ** # Plot
## Plot
time_window = np.array([1024, 1280]) / 4096
freq = np.arange(n)
L = int(np.floor(n / 2))


fig, axs = plt.subplots(2, 2)
axs = axs.reshape(-1)

axs[1].plot(freq[:L], PSD[:L], color='k', LineWidth=2)
axs[1].set_xlim(0, 1024)
axs[1].set_ylim(0, 1200)

axs[0].plot(t, x, color='k', LineWidth=2)
axs[0].plot(perm / n, y, color='r', marker='x', LineWidth=0, ms=12, mew=4)
axs[0].set_xlim(time_window[0], time_window[1])
axs[0].set_ylim(-2, 2)

axs[2].plot(t, xrecon, color='r', LineWidth=2)
axs[2].set_xlim(time_window[0], time_window[1])
axs[2].set_ylim(-2, 2)

xtrecon = np.fft.fft(xrecon, n)  # computes the (fast) discrete fourier transform
PSDrecon = xtrecon * np.conj(xtrecon) / n  # Power spectrum (how much power in each freq)

axs[3].plot(freq[:L], PSDrecon[:L], color='r', LineWidth=2)
axs[3].set_xlim(0, 1024)
axs[3].set_ylim(0, 1200)

plt.show()


#@+node:ekr.20241212100514.133: ** # L1-Minimization using SciPy
## L1-Minimization using SciPy
def L1_norm(x):
    return np.linalg.norm(x, ord=1)

constr = ({'type': 'eq', 'fun': lambda x:  Theta @x - y})
x0 = np.linalg.pinv(Theta) @y
res = minimize(L1_norm, x0, method='SLSQP', constraints=constr)
s = res.x

#@+node:ekr.20241212100514.134: ** Theta.shape
Theta.shape

#@+node:ekr.20241212100514.135: ** y.shape
y.shape

#@+node:ekr.20241212100514.136: ** Cell 9
#@-others
#@@language python
#@@tabwidth -4
#@-leo
