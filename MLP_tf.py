
# http://github.com/timestocome

# this is a very simple MLP on a very simple dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import sys



##############################################################################
# load data
# Iris data set and others available here: 
# http://archive.ics.uci.edu/ml/index.php
##############################################################################

# read in csv
data = pd.read_csv('Iris.csv')

# convert target strings to one hot vectors
data['t1'] = data['y'].apply(lambda z: 1 if z == 'Iris-setosa' else 0)
data['t2'] = data['y'].apply(lambda z: 1 if z == 'Iris-versicolor' else 0)
data['t3'] = data['y'].apply(lambda z: 1 if z == 'Iris-virginica' else 0)


# scale features
data['f1'] = (data['f1'] - data['f1'].min()) / (data['f1'].max() - data['f1'].min())
data['f2'] = (data['f2'] - data['f2'].min()) / (data['f2'].max() - data['f2'].min())
data['f3'] = (data['f3'] - data['f3'].min()) / (data['f3'].max() - data['f3'].min())
data['f4'] = (data['f4'] - data['f4'].min()) / (data['f4'].max() - data['f4'].min())


# remove string label
data = data[['f1','f2','f3','f4','t1','t2','t3']]

# shuffle data
data = data.sample(frac=1.)

# split into test and train sets
n_test = len(data) // 10
n_train = len(data) - n_test

train = data[0:n_train]
test = data[n_train:-1]

train_x = train[['f1', 'f2', 'f3', 'f4']]
train_y = train[['t1', 't2', 't3']]

test_x = test[['f1', 'f2', 'f3', 'f4']]
test_y = test[['t1', 't2', 't3']]


# convert to numpy arrays for tensorflow
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)



##############################################################################
# network
##############################################################################

lr = 0.1                # learning rate for Gradient Descent
l2 = 0.01               # alpha for regularization ( usually ~ 1/n_weights)
n_epochs = 20           # how many times to loop entire dataset
n_out = 3               # number of classes
n_features = 4          # number of input features
n_hidden = 12           # number of hidden nodes

# used as placeholders in graph, x, y input data will fill these variables
x = tf.placeholder(tf.float32, shape=[1, n_features], name='x')              # (1, 4)
y = tf.placeholder(tf.float32, shape=[1, n_out], name='y')           # (3, 1)


# variables that will be learned through training
W1 = tf.Variable(tf.random_normal([n_features, n_hidden]), name='W1')       # (12, 4)
b1 = tf.Variable(tf.ones([n_hidden]), name='b1')                            # (12,)

W2 = tf.Variable(tf.random_normal([n_hidden, n_out]), name='W2')            # (12, 3)
b2 = tf.Variable(tf.ones([n_out], name='b2'))                               # (3,)

# network equations
layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1))                        # (12, 1)
prediction = tf.nn.sigmoid(tf.add(tf.matmul(layer1, W2), b2))               # ( 3, 1)

cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction) # how did we do?
regularization = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)                      # keep weights from blowing up
loss = cost + l2 * regularization                                           # used to calculate gradients

train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)             # update weights


# tensorFlow Session - creates the graph that is the network
# and runs the equations
with tf.Session() as sess:

    # set up our variables and constants
    sess.run(tf.global_variables_initializer())

    # used to visualize tensorflow training and the graph using tensorboard
    writer = tf.summary.FileWriter('./logs', sess.graph)

    #---------------  train  ---------------------------------------------------------
    for j in range(n_epochs):

        # re-shuffle data each epoch to get more consistent results
        idx = np.arange(n_train)    # have to keep features and targets lined up
        tx = train_x[idx]
        ty = train_y[idx]

        for i in range(n_train):

            x_batch = np.reshape(tx[i], (1,4))
            y_batch = np.reshape(ty[i], (1,3))
            
            summary, cost_value, prediction_value = sess.run([train_op, cost, prediction], feed_dict={x: x_batch, y: y_batch})
            writer.add_summary(summary, i)
        
        print("Weights", np.sum(sess.run(W1)), np.sum(sess.run(W2)))
        print("Epoch: : %i, cost %f" % (j, cost_value))

    
    #------------------  test  -----------------------------------------------------
    correct = 0
    for k in range(n_test-1):

        x_batch = np.reshape(test_x[k], (1,4))
        y_batch = np.reshape(test_y[k], (1,3))

        prediction_value = sess.run([prediction], feed_dict={x: x_batch, y: y_batch})
        print(np.argmax(prediction_value), np.argmax(y_batch))
        if np.argmax(prediction_value) == np.argmax(y_batch): correct += 1

    print("Correct %d / %d" % (correct, n_test))
