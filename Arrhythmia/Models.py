#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:49:24 2018

# http://github.com/timestocome
# Linda MacPhee-Cobb


"""



##################################################################################
# data
#################################################################################
import pandas as pd
import numpy as np



x_test = pd.read_csv('x_original.csv', index_col=0)
y_test = pd.read_csv('y_original.csv', index_col=0)

'''
print('-------------------------original')
print(x_test.describe())
print(y_test.describe())
'''

n_test = len(x_test)



x = pd.read_csv('x_train.csv', index_col=0)
y = pd.read_csv('y_train.csv', index_col=0)

'''
print('-------------------------training')
print(x.describe())
print(y.describe())
'''
n_samples = len(x)
n_features = len(x.columns.values)
n_classes = len(y.columns.values)

print('total training samples', n_samples)
print('features %d, classes %d' %(n_features, n_classes))


# shuffle training data
idx = np.random.permutation(n_samples)

n_validate = n_samples // 10
n_train = n_samples - n_validate

x = x.iloc[idx]
y = y.iloc[idx]

'''
print('-----  shuffled? ----------------------------------------')
print(len(x))
print(x.head(10))

'''


# split off some validation data from training data
xt = x[0:n_train]
yt = y[0:n_train]

xv = x[n_validate:]
yv = y[n_validate:]

'''
print('n_train %d n_validate %d' %(n_samples, n_validate))
print('n train %d, n validate %d' %(len(x), len(xv)))

print(x.head(5))
print(xv.head(5))

'''




###############################################################################
# model
###############################################################################

from keras import models
from keras import layers


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(n_features, )))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(n_classes, activation='softmax'))


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



history = model.fit(xt, yt, epochs=18, batch_size=20, validation_data=(xv, yv))




model.save('arrhythmia_model.h5')


##################################################################################
# test on actual data
##############################################################################

predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)

y_test = np.argmax(y_test.values, axis=1)



correct = 0
for i in range(n_test):
    #print(predictions[i], y_test[i])
    if predictions[i] == y_test[i]:
        correct += 1
        #print('Correct:', y_test[i])
    else:
        #print('Miss:', y_test[i])
        pass
    
    
print('Prediction matches holdout', correct)
print('Accuracy', correct/len(y_test))









