#@+leo-ver=5-thin
#@+node:ekr.20241212100516.48: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH07\CH07_SEC05_HAVOK_Lorenz.py
#@+others
#@+node:ekr.20241212100516.50: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate


rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

#@+node:ekr.20241212100516.51: ** # Simulate the Lorenz System
## Simulate the Lorenz System

dt = 0.01
T = 50
t = np.arange(0, T + dt, dt)
beta = 8 / 3
sigma = 10
rho = 28


def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

np.random.seed(123)
x0 = (-8, 8, 27)

x = integrate.odeint(lorenz_deriv, x0, t, rtol=10 ** (-12), atol=10 ** (-12) * np.ones_like(x0))


#@+node:ekr.20241212100516.52: ** # Eigen-time delay coordinates
## Eigen-time delay coordinates
stackmax = 10  # Number of shift-stacked rows
r = 10  # rank of HAVOK model
H = np.zeros((stackmax, x.shape[0] - stackmax))

for k in range(stackmax):
    H[k, :] = x[k : -(stackmax - k), 0]

U, S, VT = np.linalg.svd(H, full_matrices=0)
V = VT.T

#@+node:ekr.20241212100516.53: ** # Compute Derivatives (4th Order Central
## Compute Derivatives (4th Order Central Difference)
# dV = np.zeros((V.shape[0]-5,r))
# for i in range(2,V.shape[0]-3):
#     for k in range(r):
#         dV[i-1,k] = (1/(12*dt))

dV = (1 / (12 * dt)) * (-V[4:,:] + 8 * V[3 : -1, :] - 8 * V[1 : -3, :] + V[: -4, :])

# trim first and last two that are lost in derivative
V = V[2:-2]

#@+node:ekr.20241212100516.54: ** # Build HAVOK Regression Model on Time
## Build HAVOK Regression Model on Time Delay Coordinates
Xi = np.linalg.lstsq(V, dV, rcond=None)[0]
A = Xi[: (r - 1), : (r - 1)].T
B = Xi[-1, : (r - 1)].T

#@+node:ekr.20241212100516.55: ** print(1/2/3)
print(1 / 2 / 3)

#@+node:ekr.20241212100516.56: ** Cell 7
#@-others
#@@language python
#@@tabwidth -4
#@-leo
