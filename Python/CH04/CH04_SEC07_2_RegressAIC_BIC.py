#@+leo-ver=5-thin
#@+node:ekr.20241212100515.57: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH04\CH04_SEC07_2_RegressAIC_BIC.py
#@+others
#@+node:ekr.20241212100515.59: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from statsmodels.tsa import arima_process, arima_model
# Using the StatsModels module available at
# https://www.statsmodels.org/dev/install.html


rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]


#@+node:ekr.20241212100515.60: ** for random data reproducibility
np.random.seed(123)  # for random data reproducibility
T = 100  # Sample size
# DGP = sm.ARIMA(x,order=(0,0,0))
# arparams = np.array([.2, 0.5])
# maparams = np.array([-4])
# arparams = np.r_[1, arparams]

arparams = np.array([-4, .2, 0.5])
maparams = np.array([1])



arma_process = sm.tsa.arima_process.ArmaProcess(arparams, maparams)
y = arma_process.generate_sample(T, scale=2)

logL = np.zeros(3)  # log likelihood vector
aic = np.zeros(3)  # AIC vector
bic = np.zeros(3)  # BIC vector

for j in range(2):
    model_res = sm.tsa.arima_model.ARMA(y, (0, 0)).fit(trend='c', disp=0, start_ar_lags=j + 1, method='mle')
    logL[j] = model_res.llf
    aic[j] = model_res.aic
    bic[j] = model_res.bic

print('AIC: {:}'.format(aic))
print('BIC: {:}'.format(bic))

#@+node:ekr.20241212100515.61: ** Cell 3
#@-others
#@@language python
#@@tabwidth -4
#@-leo
