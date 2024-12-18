#@+leo-ver=5-thin
#@+node:ekr.20241212100513.18: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH01\CH01_SEC04_2_Cement.py
#@+others
#@+node:ekr.20241212100513.20: ** import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

# Load dataset
A = np.loadtxt(os.path.join('..', 'DATA', 'hald_ingredients.csv'), delimiter=',')
b = np.loadtxt(os.path.join('..', 'DATA', 'hald_heat.csv'), delimiter=',')

# Solve Ax=b using SVD
U, S, VT = np.linalg.svd(A, full_matrices=0)
x = VT.T @np.linalg.inv(np.diag(S)) @U.T @b

plt.plot(b, Color='k', LineWidth=2, label='Heat Data')  # True relationship
plt.plot(A @x, '-o', Color='r', LineWidth=1.5, MarkerSize=6, label='Regression')
plt.legend()
plt.show()

#@+node:ekr.20241212100513.21: ** Alternative Methods:
# Alternative Methods:

# The first alternative is specific to Matlab:
# x = regress(b,A)

# Alternative 2:
x = np.linalg.pinv(A) * b

#@+node:ekr.20241212100513.22: ** Cell 3
#@-others
#@@language python
#@@tabwidth -4
#@-leo
