#@+leo-ver=5-thin
#@+node:ekr.20241212100514.83: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH02\CH02_SEC05_HAAR.py
#@+others
#@+node:ekr.20241212100514.85: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 18})


x = np.arange(0, 1, 0.001)
n = len(x)
n2 = int(np.floor(n / 2))
n4 = int(np.floor(n / 4))

f10 = np.zeros_like(x)
f10[:n2] = 1
f10[n2:] = -1

f21 = np.zeros_like(x)
f21[:n4] = 1
f21[n4:n2] = -1
f21 = f21 * np.sqrt(2)

f22 = np.zeros_like(x)
f22[n2 : (n2 + n4)] = 1
f22[(n2 + n4) :] = -1
f22 = f22 * np.sqrt(2)

# x = np.concatenate((-1, 0, x, 1, 2))
x = np.append([-1, 0], x)
x = np.append(x, [1, 2])

f10 = np.pad(f10, (2, 2), 'constant')
f21 = np.pad(f21, (2, 2), 'constant')
f22 = np.pad(f22, (2, 2), 'constant')

fig, axs = plt.subplots(3, 1)
axs[0].plot(x, f10, color='k', LineWidth=2)
axs[0].set_xlim(-0.2, 1.2)
axs[0].set_ylim(-1.75, 1.75)
axs[1].plot(x, f21, color='k', LineWidth=2)
axs[1].set_xlim(-0.2, 1.2)
axs[1].set_ylim(-1.75, 1.75)
axs[2].plot(x, f22, color='k', LineWidth=2)
axs[2].set_xlim(-0.2, 1.2)
axs[2].set_ylim(-1.75, 1.75)
plt.show()

#@+node:ekr.20241212100514.86: ** x = np.arange(-5,5,0.001)
x = np.arange(-5, 5, 0.001)
fMexHat = (1 - np.power(x, 2)) * np.exp(-np.power(x, 2) / 2)
plt.plot(x, fMexHat, color='k', LineWidth=2)
plt.show()

#@+node:ekr.20241212100514.87: ** Cell 3
#@-others
#@@language python
#@@tabwidth -4
#@-leo
