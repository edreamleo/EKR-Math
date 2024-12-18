#@+leo-ver=5-thin
#@+node:ekr.20241212100515.149: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH06\CH06_SEC01_1_NN.py
#@+others
#@+node:ekr.20241212100515.151: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import io
import os
from sklearn import linear_model


rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

#@+node:ekr.20241212100515.152: ** dogs vs. cats
# dogs vs. cats
dogdata_w_mat = io.loadmat(os.path.join('..', 'DATA', 'dogData_w.mat'))
catdata_w_mat = io.loadmat(os.path.join('..', 'DATA', 'catData_w.mat'))

dog_wave = dogdata_w_mat['dog_wave']
cat_wave = catdata_w_mat['cat_wave']

CD = np.concatenate((dog_wave, cat_wave), axis=1)

train = np.concatenate((dog_wave[:,:60], cat_wave[:,:60]), axis=1)
test = np.concatenate((dog_wave[:, 60 : 80], cat_wave[:, 60 : 80]), axis=1)
label = np.repeat(np.array([1, -1]), 60)

A = label @np.linalg.pinv(train)
test_labels = np.sign(A @test)

#@+node:ekr.20241212100515.153: ** lasso =
lasso = linear_model.Lasso().fit(train.T, label)
A_lasso = lasso.coef_
test_labels_lasso = np.sign(A_lasso @test)

#@+node:ekr.20241212100515.154: ** fig,axs = plt.subplots(4,1)
fig, axs = plt.subplots(4, 1)
axs[0].bar(range(len(test_labels)), test_labels)
axs[1].bar(range(len(A)), A)
axs[2].bar(range(len(test_labels_lasso)), test_labels_lasso)
axs[3].bar(range(len(A_lasso)), A_lasso)


plt.show()

#@+node:ekr.20241212100515.155: ** fig,axs = plt.subplots(2,2)
fig, axs = plt.subplots(2, 2)
axs = axs.reshape(-1)
A2 = np.flipud(np.reshape(A, (32, 32)))
A2_lasso = np.flipud(np.reshape(A_lasso, (32, 32)))
axs[0].pcolor(np.rot90(A2), cmap='gray')
axs[1].pcolor(np.rot90(A2_lasso), cmap='gray')


plt.show()

#@+node:ekr.20241212100515.156: ** # To be implemented: Python version of
## To be implemented: Python version of Matlab's patternnet()

#@+node:ekr.20241212100515.157: ** Cell 7
#@+node:ekr.20241212100515.158: ** Cell 8
#@-others
#@@language python
#@@tabwidth -4
#@-leo
