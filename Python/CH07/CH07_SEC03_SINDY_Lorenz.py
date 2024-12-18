#@+leo-ver=5-thin
#@+node:ekr.20241212100516.32: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH07\CH07_SEC03_SINDY_Lorenz.py
#@+others
#@+node:ekr.20241212100516.34: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate


rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

#@+node:ekr.20241212100516.35: ** # Simulate the Lorenz System
## Simulate the Lorenz System

dt = 0.01
T = 50
t = np.arange(dt, T + dt, dt)
beta = 8 / 3
sigma = 10
rho = 28
n = 3

def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

np.random.seed(123)
x0 = (-8, 8, 27)

x = integrate.odeint(lorenz_deriv, x0, t, rtol=10 ** (-12), atol=10 ** (-12) * np.ones_like(x0))


#@+node:ekr.20241212100516.36: ** # Compute Derivative
## Compute Derivative
dx = np.zeros_like(x)
for j in range(len(t)):
    dx[j, :] = lorenz_deriv(x[j, :], 0, sigma, beta, rho)


#@+node:ekr.20241212100516.37: ** # SINDy Function Definitions
## SINDy Function Definitions

def poolData(yin, nVars, polyorder):
    n = yin.shape[0]
    yout = np.zeros((n, 1))

    # poly order 0
    yout[:, 0] = np.ones(n)

    # poly order 1
    for i in range(nVars):
        yout = np.append(yout, yin[:, i].reshape((yin.shape[0], 1)), axis=1)

    # poly order 2
    if polyorder >= 2:
        for i in range(nVars):
            for j in range(i, nVars):
                yout = np.append(yout, (yin[:, i] * yin[:, j]).reshape((yin.shape[0], 1)), axis=1)

    # poly order 3
    if polyorder >= 3:
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    yout = np.append(yout, (yin[:, i] * yin[:, j] * yin[:, k]).reshape((yin.shape[0], 1)), axis=1)

    return yout

def sparsifyDynamics(Theta, dXdt, lamb, n):
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]  # Initial guess: Least-squares

    for k in range(10):
        smallinds = np.abs(Xi) < lamb  # Find small coefficients
        Xi[smallinds] = 0  # and threshold
        for ind in range(n):  # n is state dimension
            biginds = smallinds[:, ind] == 0
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], dXdt[:, ind], rcond=None)[0]

    return Xi


#@+node:ekr.20241212100516.38: ** Up to third order polynomials
Theta = poolData(x, n, 3)  # Up to third order polynomials
lamb = 0.025  # sparsification knob lambda
Xi = sparsifyDynamics(Theta, dx, lamb, n)

print(Xi)

#@+node:ekr.20241212100516.39: ** Cell 6
#@-others
#@@language python
#@@tabwidth -4
#@-leo
