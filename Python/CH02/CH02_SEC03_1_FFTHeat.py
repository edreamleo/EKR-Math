#@+leo-ver=5-thin
#@+node:ekr.20241212100514.66: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH02\CH02_SEC03_1_FFTHeat.py
#@+others
#@+node:ekr.20241212100514.68: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})

a = 1  # Thermal diffusivity constant
L = 100  # Length of domain
N = 1000  # Number of discretization points
dx = L / N
x = np.arange(-L / 2, L / 2, dx)  # Define x domain

# Define discrete wavenumbers
kappa = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# Initial condition
u0 = np.zeros_like(x)
u0[int((L / 2 - L / 10) / dx) : int((L / 2 + L / 10) / dx)] = 1
u0hat = np.fft.fft(u0)

# SciPy's odeint function doesn't play well with complex numbers, so we recast
# the state u0hat from an N-element complex vector to a 2N-element real vector
u0hat_ri = np.concatenate((u0hat.real, u0hat.imag))

# Simulate in Fourier frequency domain
dt = 0.1
t = np.arange(0, 10, dt)

def rhsHeat(uhat_ri, t, kappa, a):
    uhat = uhat_ri[:N] + (1j) * uhat_ri[N:]
    d_uhat = -a ** 2 * (np.power(kappa, 2)) * uhat
    d_uhat_ri = np.concatenate((d_uhat.real, d_uhat.imag)).astype('float64')
    return d_uhat_ri

uhat_ri = odeint(rhsHeat, u0hat_ri, t, args=(kappa, a))

uhat = uhat_ri[:,:N] + (1j) * uhat_ri[:, N :]

u = np.zeros_like(uhat)

for k in range(len(t)):
    u[k, :] = np.fft.ifft(uhat[k, :])

u = u.real

# Waterfall plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u_plot = u[0 : -1 : 10, :]
for j in range(u_plot.shape[0]):
    ys = j * np.ones(u_plot.shape[1])
    ax.plot(x, ys, u_plot[j, :])

# Image plot
plt.figure()
plt.imshow(np.flipud(u), aspect=8)
plt.axis('off')
plt.show()

#@+node:ekr.20241212100514.69: ** Cell 2
#@-others
#@@language python
#@@tabwidth -4
#@-leo
