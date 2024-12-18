#@+leo-ver=5-thin
#@+node:ekr.20241212100514.194: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH04\CH04_SEC01_LinearRegression.py
#@+others
#@+node:ekr.20241212100514.196: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from matplotlib import rcParams
rcParams['figure.figsize'] = [12, 8]
rcParams.update({'font.size': 18})


# Function definitions
def fit1(x0, t):
    x, y = t
    return np.max(np.abs(x0[0] * x + x0[1] - y))
def fit2(x0, t):
    x, y = t
    return np.sum(np.abs(x0[0] * x + x0[1] - y))
def fit3(x0, t):
    x, y = t
    return np.sum(np.power(np.abs(x0[0] * x + x0[1] - y), 2))


#@+node:ekr.20241212100514.197: ** The data
# The data
x = np.arange(1, 11)
y = np.array([0.2, 0.5, 0.3, 3.5, 1.0, 1.5, 1.8, 2.0, 2.3, 2.2])
t = (x, y)

x0 = np.array([1, 1])
p1 = scipy.optimize.fmin(fit1, x0, args=(t,));
p2 = scipy.optimize.fmin(fit2, x0, args=(t,));
p3 = scipy.optimize.fmin(fit3, x0, args=(t,));

xf = np.arange(0, 11, 0.1)
y1 = np.polyval(p1, xf)
y2 = np.polyval(p2, xf)
y3 = np.polyval(p3, xf)

plt.figure()
plt.plot(xf, y1, color='k', label='E_\infty')
plt.plot(xf, y2, '--', color='k', LineWidth=2, label='E_1')
plt.plot(xf, y3, color='k', LineWidth=2, label='E_2')
plt.plot(x, y, 'o', color='r', LineWidth=2)

plt.ylim(0, 4)
plt.legend()
plt.show()

#@+node:ekr.20241212100514.198: ** x = np.arange(1,11)
x = np.arange(1, 11)
y = np.array([0.2, 0.5, 0.3, 0.7, 1.0, 1.5, 1.8, 2.0, 2.3, 2.2])
t = (x, y)

x0 = np.array([1, 1])
p1 = scipy.optimize.fmin(fit1, x0, args=(t,));
p2 = scipy.optimize.fmin(fit2, x0, args=(t,));
p3 = scipy.optimize.fmin(fit3, x0, args=(t,));

xf = np.arange(0, 11, 0.1)
y1 = np.polyval(p1, xf)
y2 = np.polyval(p2, xf)
y3 = np.polyval(p3, xf)

plt.figure()
plt.plot(xf, y1, color='k', label='E_\infty')
plt.plot(xf, y2, '--', color='k', LineWidth=2, label='E_1')
plt.plot(xf, y3, color='k', LineWidth=2, label='E_2')
plt.plot(x, y, 'o', color='r', LineWidth=2)

plt.ylim(0, 4)
plt.legend()
plt.show()

#@+node:ekr.20241212100514.199: ** Cell 4
#@+node:ekr.20241212100514.200: ** Cell 5
#@-others
#@@language python
#@@tabwidth -4
#@-leo
