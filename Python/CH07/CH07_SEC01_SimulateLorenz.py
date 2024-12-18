#@+leo-ver=5-thin
#@+node:ekr.20241212100516.16: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH07\CH07_SEC01_SimulateLorenz.py
#@+others
#@+node:ekr.20241212100516.18: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate


rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

#@+node:ekr.20241212100516.19: ** # Simulate the Lorenz System
## Simulate the Lorenz System

dt = 0.001
T = 50
t = np.arange(0, T + dt, dt)
beta = 8 / 3
sigma = 10
rho = 28

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})


def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

np.random.seed(123)
x0 = (0, 1, 20)

x_t = integrate.odeint(lorenz_deriv, x0, t, rtol=10 ** (-12), atol=10 ** (-12) * np.ones_like(x0))

x, y, z = x_t.T
plt.plot(x, y, z, linewidth=1)
plt.scatter(x0[0], x0[1], x0[2], color='r')

ax.view_init(18, -113)
plt.show()

#@+node:ekr.20241212100516.20: ** Cell 3
#@+node:ekr.20241212100516.21: ** Cell 4
#@-others
#@@language python
#@@tabwidth -4
#@-leo
