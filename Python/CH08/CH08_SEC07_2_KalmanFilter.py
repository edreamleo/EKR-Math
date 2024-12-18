#@+leo-ver=5-thin
#@+node:ekr.20241212100516.83: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH08\CH08_SEC07_2_KalmanFilter.py
#@+others
#@+node:ekr.20241212100516.85: ** import numpy as np
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

#@+node:ekr.20241212100516.86: ** m = 1
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

C = np.array([1, 0, 0, 0])  # measure cart position, x

print('Observability matrix rank: {}'.format(np.linalg.matrix_rank(obsv(A, C))))

D = np.zeros((C.shape[0], B.shape[1]))


#@+node:ekr.20241212100516.87: ** Kalman estimator design
# Kalman estimator design

def lqe(a, g, c, q, r):
    r = np.atleast_2d(r)
    nn = np.zeros((q.shape[0], len(r)))
    qg = g @q @g.T
    ng = g @nn

    qg = (qg + qg.T) / 2
    r = (r + r.T) / 2
    u, t = schur(r)

    t = np.real(np.diag(t))

    if np.min(t) <= 0:
        print('Error: covariance matrix must be positive definite')
    else:
        Nr = (ng @u) * np.diag(np.power(np.sqrt(t), -1))
        Qr = qg - Nr @Nr.T
        if np.min(np.real(np.linalg.eig(Qr)[0])) < -(10 ** 3) * np.finfo(float).eps:
            print('Warning: The matrix [G*Q*G' ' G*N;N' '*G' ' R] should be nonnegative definite')
    c = np.diag(c)
    r = np.squeeze(r)
    (p, e, k) = care(a.T, c.T, qg)  #,R=r,S=ng)
    l = k[0, :]

    return (l, p, e)


#@+node:ekr.20241212100516.88: ** # Specify disturbance and noise
## Specify disturbance and noise magnitude
Vd = np.eye(4)  # distrubance covariance
Vn = 1  # noise covariance

# Build Kalman filter
Kf, P, E = lqe(A, np.eye(4), C, Vd, Vn)


#@+node:ekr.20241212100516.89: ** # Augment system with additional inputs
## Augment system with additional inputs
Baug = np.concatenate((B, np.eye(4), np.zeros_like(B)), axis=1)  # [u I*wd 0*wn]
Daug = np.array([0, 0, 0, 0, 0, 1])  # D matrix passes noise through

sysC = ss(A, Baug, C, Daug)  # Single-measurement system

# "True" system w/ full-state output, disturbance, no noise
sysTruth = ss(A, Baug, np.eye(4), np.zeros((4, Baug.shape[1])))

BKf = np.concatenate((B, np.atleast_2d(Kf).T), axis=1)
sysKF = ss(A - np.outer(Kf, C), BKf, np.eye(4), np.zeros_like(BKf))

#@+node:ekr.20241212100516.90: ** # Estimate linearized system in "down"
## Estimate linearized system in "down" position (Gantry crane)
dt = 0.01
t = np.arange(0, 50, dt)

uDIST = np.sqrt(Vd) @np.random.randn(4, len(t))  # random disturbance
uNOISE = np.sqrt(Vn) * np.random.randn(len(t))  # random noise
u = np.zeros_like(t)
u[100] = 20 / dt  # positive impulse
u[1500] = -20 / dt  # negative impulse

# input w/ disturbance and noise:
uAUG = np.concatenate((u.reshape((1, len(u))), uDIST, uNOISE.reshape((1, len(uNOISE))))).T

y, t, _ = lsim(sysC, uAUG, t)  # noisy measurement
xtrue, t, _ = lsim(sysTruth, uAUG, t)  # true state
xhat, t, _ = lsim(sysKF, np.row_stack((u, y)).T, t)  # state estimate

#@+node:ekr.20241212100516.91: ** plt.plot(t,y,color=(0.5,0.5,0.5),label='
plt.plot(t, y, color=(0.5, 0.5, 0.5), label='y (measured)')
plt.plot(t, xtrue[:, 0], color='k', label='y (no noise)')
plt.plot(t, xhat[:, 0], color=(0, 0.447, 0.741), label='y (KF estimate)')
plt.legend()
plt.show()

#@+node:ekr.20241212100516.92: ** x_labels = ('x','v','theta','omega')
x_labels = ('x', 'v', 'theta', 'omega')
[plt.plot(t, xtrue[:, k], linewidth=1.2, label=x_labels[k]) for k in range(4)]
plt.gca().set_prop_cycle(None)  # reset color cycle
[plt.plot(t, xhat[:, k], '--', linewidth=2, label=x_labels[k] + '_hat') for k in range(4)]
plt.legend()
plt.show()

#@+node:ekr.20241212100516.93: ** Cell 9
#@-others
#@@language python
#@@tabwidth -4
#@-leo
