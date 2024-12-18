#@+leo-ver=5-thin
#@+node:ekr.20241212100515.32: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH04\CH04_SEC05_0_Fig4p16_Pareto.py
#@+others
#@+node:ekr.20241212100515.34: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams
rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [8, 8]


#@+node:ekr.20241212100515.35: ** fig,ax = plt.subplots(1)
fig, ax = plt.subplots(1)
x = np.arange(0.2, 5, 0.1)
y = np.divide(1, x)

x2 = np.copy(x)
n = len(x2)
y2 = np.divide(1, x2) + 0.5 * np.random.randn(n)

y3 = (np.tile(y2, (5, 1)) + 2 * np.random.rand(5, n) + 1)
y3 = np.reshape(y3, -1)
x3 = np.tile(x, (1, 5)).reshape(-1)

plt.plot(x, y, color='k', LineWidth=2)

rect = Rectangle((0.5, 0.4), 0.9, 1.6, LineWidth=1, edgecolor='k', facecolor='grey', alpha=0.6)
ax.add_patch(rect)

plt.scatter(x2, y2, 100, color='magenta', edgecolors='k')
plt.scatter(x3, y3, 100, color='lime', edgecolors='k')


plt.xlim(0.2, 4)
plt.ylim(0, 5.5)

plt.show()

#@+node:ekr.20241212100515.36: ** Cell 3
#@-others
#@@language python
#@@tabwidth -4
#@-leo
