#@+leo-ver=5-thin
#@+node:ekr.20241212100514.59: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH02\CH02_SEC02_3_SpectralDerivative.py
#@+others
#@+node:ekr.20241212100514.61: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})


n = 128
L = 30
dx = L / n
x = np.arange(-L / 2, L / 2, dx, dtype='complex_')
f = np.cos(x) * np.exp(-np.power(x, 2) / 25)  # Function
df = -(np.sin(x) * np.exp(-np.power(x, 2) / 25) + (2 / 25) * x * f)  # Derivative

#@+node:ekr.20241212100514.62: ** # Approximate derivative using finite
## Approximate derivative using finite difference
dfFD = np.zeros(len(df), dtype='complex_')
for kappa in range(len(df) - 1):
    dfFD[kappa] = (f[kappa + 1] - f[kappa]) / dx

dfFD[-1] = dfFD[-2]

#@+node:ekr.20241212100514.63: ** # Derivative using FFT (spectral
## Derivative using FFT (spectral derivative)
fhat = np.fft.fft(f)
kappa = (2 * np.pi / L) * np.arange(-n / 2, n / 2)
kappa = np.fft.fftshift(kappa)  # Re-order fft frequencies
dfhat = kappa * fhat * (1j)
dfFFT = np.real(np.fft.ifft(dfhat))


#@+node:ekr.20241212100514.64: ** # Plots
## Plots
plt.plot(x, df.real, color='k', LineWidth=2, label='True Derivative')
plt.plot(x, dfFD.real, '--', color='b', LineWidth=1.5, label='Finite Diff.')
plt.plot(x, dfFFT.real, '--', color='r', LineWidth=1.5, label='FFT Derivative')
plt.legend()
plt.show()

#@+node:ekr.20241212100514.65: ** Cell 5
#@-others
#@@language python
#@@tabwidth -4
#@-leo
