#@+leo-ver=5-thin
#@+node:ekr.20241212100516.77: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH08\CH08_SEC07_2b_Obsv.py
#@+others
#@+node:ekr.20241212100516.79: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from control.matlab import *
import slycot
from scipy import integrate
from scipy.linalg import schur
# Python control toolbox available at https://python-control.readthedocs.io/

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

#@+node:ekr.20241212100516.80: ** m = 1
m = 1
M = 5
L = 2
g = -10
d = 1

b = -1  # pendulum down (b = -1)

A = np.array([[0, 1, 0, 0],\
              [0, -d / M, b * m * g / M, 0],\
              [0, 0, 0, 1],\
              [0, -b * d / (M * L), -b * (m + M) * g / (M * L), 0]])

B = np.array([0, 1 / M, 0, b / (M * L)]).reshape((4, 1))

C = np.array([0, 0, 1, 0])  # only observable if x measured... because x can't be

print('Observability matrix:\n{}'.format(obsv(A, C)))
print('Observability matrix determinant: {}'.format(np.linalg.det(obsv(A, C))))


#@+node:ekr.20241212100516.81: ** # Which measurements are best if we omit
## Which measurements are best if we omit "x"
Ah = A[1 :, 1 :]
Bh = B[1:]
# Ch = np.array([1,0,0])
Ch = np.array([0, 1, 0])
# Ch = np.array([0,0,1])

print('Observability matrix:\n{}'.format(obsv(Ah, Ch)))

Ch = Ch.reshape((1, len(Ch)))
Dh = np.zeros((Ch.shape[0], Bh.shape[1]))
sys = ss(Ah, Bh, Ch, Dh)
print('Gramian determinant: {}'.format(np.linalg.det(gram(sys, 'o'))))


#@+node:ekr.20241212100516.82: ** Cell 4
#@-others
#@@language python
#@@tabwidth -4
#@-leo
