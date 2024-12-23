#@+leo-ver=5-thin
#@+node:ekr.20241212100513.45: * @file Python/CH01\CH01_SEC07_1.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

#@+others
#@+node:ekr.20241212100513.47: ** Plot
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

t = np.arange(-3, 3, 0.01)

Utrue = np.array([np.cos(17 * t) * np.exp(-t ** 2), np.sin(11 * t)]).T
Strue = np.array([[2, 0], [0, 0.5]])
Vtrue = np.array([np.sin(5 * t) * np.exp(-t ** 2), np.cos(13 * t)]).T

X = Utrue @Strue @Vtrue.T

plt.imshow(X)
plt.set_cmap('gray')
plt.axis('off')
plt.show()

#@+node:ekr.20241212100513.48: ** Plot
sigma = 1
Xnoisy = X + sigma * np.random.randn(*X.shape)
plt.imshow(Xnoisy)
plt.set_cmap('gray')
plt.axis('off')
plt.show()

#@+node:ekr.20241212100513.49: ** Plot
U, S, VT = np.linalg.svd(Xnoisy, full_matrices=0)
N = Xnoisy.shape[0]
cutoff = (4 / np.sqrt(3)) * np.sqrt(N) * sigma  # Hard threshold
r = np.max(np.where(S > cutoff))  # Keep modes w/ sig > cutoff

Xclean = U[:, : (r + 1)] @np.diag(S[: (r + 1)]) @VT[: (r + 1), :]
plt.imshow(Xclean)
plt.set_cmap('gray')
plt.axis('off')
plt.show()

#@+node:ekr.20241212100513.50: ** Cumulative energy
cdS = np.cumsum(S) / np.sum(S)  # Cumulative energy
r90 = np.min(np.where(cdS > 0.90))  # Find r to capture 90% energy

X90 = U[:, : (r90 + 1)] @np.diag(S[: (r90 + 1)]) @VT[: (r90 + 1), :]
plt.imshow(X90)
plt.set_cmap('gray')
plt.axis('off')
plt.show()

#@+node:ekr.20241212100513.51: ** Singular Values
## Plot Singular Values

fig1, ax1 = plt.subplots(1)

ax1.semilogy(S, '-o', color='k', linewidth=2)
ax1.semilogy(np.diag(S[: (r + 1)]), 'o', color='r', linewidth=2)
ax1.plot(np.array([-20, N + 20]), np.array([cutoff, cutoff]), '--', color='r', linewidth=2)
rect = patches.Rectangle((-5, 20), 100, 200, linewidth=2, linestyle='--', edgecolor='k')  ### FaceColor='none',
ax1.add_patch(rect)
plt.xlim((-10, 610))
plt.ylim((0.003, 300))
ax1.grid()
plt.show()

fig2, ax2 = plt.subplots(1)

ax2.semilogy(S, '-o', color='k', linewidth=2)
ax2.semilogy(np.diag(S[: (r + 1)]), 'o', color='r', linewidth=2)
ax2.plot(np.array([-20, N + 20]), np.array([cutoff, cutoff]), '--', color='r', linewidth=2)
plt.xlim((-5, 100))
plt.ylim((20, 200))
ax2.grid()
plt.show()

fig3, ax3 = plt.subplots(1)
ax3.plot(cdS, '-o', color='k', linewidth=2)
ax3.plot(cdS[: (r90 + 1)], 'o', color='b', linewidth=2)
ax3.plot(cdS[: (r + 1)], 'o', color='r', linewidth=2)
plt.xticks(np.array([0, 300, r90, 600]))
plt.yticks(np.array([0, 0.5, 0.9, 1]))
plt.xlim((-10, 610))
ax3.plot(np.array([r90, r90, -10]), np.array([0, 0.9, 0.9]), '--', color='b', linewidth=2)

ax3.grid()
plt.show()



#@-others
#@@language python
#@@tabwidth -4
#@-leo
