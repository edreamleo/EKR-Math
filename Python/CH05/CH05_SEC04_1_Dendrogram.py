#@+leo-ver=5-thin
#@+node:ekr.20241212100515.98: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH05\CH05_SEC04_1_Dendrogram.py
#@+others
#@+node:ekr.20241212100515.100: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

#@+node:ekr.20241212100515.101: ** Training and testing set sizes
# Training and testing set sizes
n1 = 100  # Train
n2 = 50  # Test

# Random ellipse 1 centered at (0,0)
x = np.random.randn(n1 + n2)
y = 0.5 * np.random.randn(n1 + n2)

# Random ellipse 2 centered at (1,-2)
x2 = np.random.randn(n1 + n2) + 1
y2 = 0.2 * np.random.randn(n1 + n2) - 2

# Rotate ellipse 2 by theta
theta = np.pi / 4
A = np.zeros((2, 2))
A[0, 0] = np.cos(theta)
A[0, 1] = -np.sin(theta)
A[1, 0] = np.sin(theta)
A[1, 1] = np.cos(theta)

x3 = A[0, 0] * x2 + A[0, 1] * y2
y3 = A[1, 0] * x2 + A[1, 1] * y2

#@+node:ekr.20241212100515.102: ** plt.figure()
plt.figure()
plt.plot(x[:n1], y[:n1], 'ro')
plt.plot(x3[:n1], y3[:n1], 'bo')
plt.show()

#@+node:ekr.20241212100515.103: ** Training set: first 200 of 240 points
# Training set: first 200 of 240 points
X1 = np.column_stack((x3[:n1], y3[:n1]))
X2 = np.column_stack((x[:n1], y[:n1]))

Y = np.concatenate((X1, X2))
Z = np.column_stack((np.ones(n1), 2 * np.ones(n1)))

# Test set: remaining 40 points
x1test = np.column_stack((x3[n1:], y3[n1:]))
x2test = np.column_stack((x[n1:], y[n1:]))


#@+node:ekr.20241212100515.104: ** # Dendrograms
## Dendrograms

Y3 = np.concatenate((X1[: 50, :], X2[: 50, :]))

Y2 = pdist(Y3, metric='euclidean')
Z = hierarchy.linkage(Y2, method='average')
thresh = 0.85 * np.max(Z[:, 2])

plt.figure()
dn = hierarchy.dendrogram(Z, p=100, color_threshold=thresh)
plt.axis('off')

plt.show()

#@+node:ekr.20241212100515.105: ** plt.bar(range(100),dn['leaves'])
plt.bar(range(100), dn['leaves'])
plt.plot(np.array([0, 100]), np.array([50, 50]), 'r:', LineWidth=2)
plt.plot(np.array([50.5, 50.5]), np.array([0, 100]), 'r:', LineWidth=2)

plt.show()

#@+node:ekr.20241212100515.106: ** thresh = 0.25*np.max(Z[:,2])
thresh = 0.25 * np.max(Z[:, 2])

plt.figure()
dn = hierarchy.dendrogram(Z, p=100, color_threshold=thresh)
plt.axis('off')
plt.show()

#@+node:ekr.20241212100515.107: ** Cell 8
#@+node:ekr.20241212100515.108: ** Cell 9
#@-others
#@@language python
#@@tabwidth -4
#@-leo
