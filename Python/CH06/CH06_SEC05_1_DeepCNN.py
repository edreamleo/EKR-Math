#@+leo-ver=5-thin
#@+node:ekr.20241212100515.167: * @file C:\Users\Dev\EKR-Study\python\CODE_PYTHON\CH06\CH06_SEC05_1_DeepCNN.py
#@+others
#@+node:ekr.20241212100515.169: ** import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import io
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras import optimizers


rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

#@+node:ekr.20241212100515.170: ** letters_train_mat = io.loadmat(os.path.j
letters_train_mat = io.loadmat(os.path.join('..', 'DATA', 'lettersTrainSet.mat'))
letters_test_mat = io.loadmat(os.path.join('..', 'DATA', 'lettersTestSet.mat'))
XTrain = letters_train_mat['XTrain']
TTrain = letters_train_mat['TTrain_cell']
XTest = letters_test_mat['XTest']
TTest = letters_test_mat['TTest_cell']

perm = np.random.permutation(1500)[:20]


# By default, Keras expects data in form (batch, height, width, channels)
XTrain = np.transpose(XTrain, axes=[3, 0, 1, 2])
XTest = np.transpose(XTest, axes=[3, 0, 1, 2])



fig, axs = plt.subplots(4, 5)
axs = axs.reshape(-1)

for j in range(len(axs)):
    axs[j].imshow(np.squeeze(XTrain[perm[j], :, :, :]), cmap='gray')
    axs[j].axis('off')

#@+node:ekr.20241212100515.171: ** classes = np.unique(TTrain)
classes = np.unique(TTrain)
y_train_label = np.zeros_like(TTrain)
y_test_label = np.zeros_like(TTest)
for nc in range(len(classes)):
    y_train_label[TTrain == classes[nc]] = nc
    y_test_label[TTest == classes[nc]] = nc

y_train_label = y_train_label.reshape(-1)
y_test_label = y_test_label.reshape(-1)

# one-hot encode categorical classes
y_train = to_categorical(y_train_label)
y_test = to_categorical(y_test_label)

#@+node:ekr.20241212100515.172: ** create model
# create model
model = Sequential()

# add model layers
model.add(Conv2D(filters=16, kernel_size=5, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(len(classes), activation='softmax'))

sgd_optimizer = optimizers.SGD(momentum=0.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
model.fit(XTrain, y_train, epochs=30)

#@+node:ekr.20241212100515.173: ** YPredict =
YPredict = np.argmax(model.predict(XTest), axis=1)
# argmax reverses the one-hot encoding scheme

accuracy = np.sum(YPredict == y_test_label) / len(y_test_label)
print('Accuracy = {}'.format(accuracy))

#@+node:ekr.20241212100515.174: ** Cell 6
#@+node:ekr.20241212100515.175: ** Cell 7
#@-others
#@@language python
#@@tabwidth -4
#@-leo
