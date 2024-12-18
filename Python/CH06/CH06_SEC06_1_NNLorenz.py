#@+leo-ver=5-thin
#@+node:ekr.20241212100516.1: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH06\CH06_SEC06_1_NNLorenz.py
#@+others
#@+node:ekr.20241212100516.3: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras import optimizers
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K


rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

#@+node:ekr.20241212100516.4: ** # Simulate the Lorenz System
## Simulate the Lorenz System

dt = 0.01
T = 8
t = np.arange(0, T + dt, dt)
beta = 8 / 3
sigma = 10
rho = 28


nn_input = np.zeros((100 * (len(t) - 1), 3))
nn_output = np.zeros_like(nn_input)

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})


def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

np.random.seed(123)
x0 = -15 + 30 * np.random.random((100, 3))

x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t)
                  for x0_j in x0])

for j in range(100):
    nn_input[j * (len(t) - 1) : (j + 1) * (len(t) - 1), :] = x_t[j, : -1, :]
    nn_output[j * (len(t) - 1) : (j + 1) * (len(t) - 1), :] = x_t[j, 1 :, :]
    x, y, z = x_t[j, :, :].T
    ax.plot(x, y, z, linewidth=1)
    ax.scatter(x0[j, 0], x0[j, 1], x0[j, 2], color='r')

ax.view_init(18, -113)
plt.show()


#@+node:ekr.20241212100516.5: ** # Neural Net
## Neural Net

# Define activation functions
def logsig(x):
    return K.variable(np.divide(1, (1 + np.exp(-K.eval(x)))))

def radbas(x):
    return K.variable(np.exp(-np.power(K.eval(x), 2)))

def purelin(x):
    return x


# create model
model = Sequential()

# add model layers
model.add(Dense(10, activation=logsig))
model.add(Dense(10, activation=radbas))
model.add(Dense(10, activation=purelin))




sgd_optimizer = optimizers.SGD(momentum=0.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
model.fit(nn_input, nn_output, epochs=30)

#@+node:ekr.20241212100516.6: ** nn_input.shape
nn_input.shape

#@+node:ekr.20241212100516.7: ** Cell 5
#@-others
#@@language python
#@@tabwidth -4
#@-leo
