#@+leo-ver=5-thin
#@+node:ekr.20241212100516.58: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH08\CH08_SEC01_CruiseControl.py
#@+others
#@+node:ekr.20241212100516.60: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

#@+node:ekr.20241212100516.61: ** time
t = np.arange(0, 10, 0.01)  # time

wr = 60 * np.ones_like(t)  # reference speed
d = 10 * np.sin(np.pi * t)  # disturbance

aModel = 1  # y = aModel*u
aTrue = 0.5  # y = aTrue*u

uOL = wr / aModel  # Open-loop u based on model
yOL = aTrue * uOL + d  # Open-loop response

K = 50  # control gain, u=K(wr-y)
yCL = (aTrue * K / (1 + aTrue * K)) * wr + d / (1 + aTrue * K)

#@+node:ekr.20241212100516.62: ** plt.plot(t,wr,'k',linewidth=2,label='Ref
plt.plot(t, wr, 'k', linewidth=2, label='Reference')
plt.plot(t, d, 'k--', linewidth=1.5, label='Disturbance')
plt.plot(t, yOL, 'r', linewidth=1.5, label='Open Loop')
plt.plot(t, yCL, 'b', linewidth=1.5, label='Closed Loop')

plt.xlabel('Time')
plt.ylabel('Speed')

plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.show()

#@+node:ekr.20241212100516.63: ** Cell 4
#@-others
#@@language python
#@@tabwidth -4
#@-leo
