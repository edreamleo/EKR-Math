#@+leo-ver=5-thin
#@+node:ekr.20241212100514.78: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH02\CH02_SEC04_1_SpectrogramChirp.py
#@+others
#@+node:ekr.20241212100514.80: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 18})

dt = 0.001
t = np.arange(0, 2, dt)
f0 = 50
f1 = 250
t1 = 2
x = np.cos(2 * np.pi * t * (f0 + (f1 - f0) * np.power(t, 2) / (3 * t1 ** 2)))

plt.specgram(x, NFFT=128, Fs=1 / dt, noverlap=120, cmap='jet')
plt.colorbar()
plt.show()

#@+node:ekr.20241212100514.81: ** Cell 2
#@+node:ekr.20241212100514.82: ** Cell 3
#@-others
#@@language python
#@@tabwidth -4
#@-leo
