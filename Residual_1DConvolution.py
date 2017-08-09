
# http://github.com/timestocome


# try a simplified version of Ng's arrhythmia detection network
# While the data is too weighted ( half of the targets = 1)
# and the dataset is very small.
# This network gets about half correct 
# The MLP I tested it on only guessed 1's This guesses multiple classes



# cardiologist level arrhythmia detection with deep CNN
# https://arxiv.org/pdf/1707.01836.pdf

# deep residual learning for image recognition
# https://arxiv.org/pdf/1512.03385.pdf



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import sys



##############################################################################
# load data
# original data downloaded here:
# http://archive.ics.uci.edu/ml/datasets/Arrhythmia
# 
# data cleaning code is CleanArrhythmia.py
# using only the 2nd signal to test the network
##############################################################################


# read in files created in CleanArrhythmia.py
misc_data = pd.read_csv('Arr_misc.csv')     # age, bmi, ... 
signal_1 = pd.read_csv('Arr_signal1.csv')   
signal_2 = pd.read_csv('Arr_signal2.csv')


#print(misc_data.shape)
#print(signal_1.shape)
#print(signal_2.shape)

#print(signal_2)
# use signal_2 to test the network
data = signal_2

# get unique values to build a one hot array for targets
# and check data is balanced
# the data is heavily weighted with 1 being 245 of the 452 target values
#     so unlikely network will perform well, but it'll still be good for testing
#     network structure
#print(data.Target.min(), data.Target.max())
#print(data.Target.unique())
#print(data.Target.value_counts())

# targets = 1:16
targets_array = np.zeros((452, 17))

for index, row in signal_2.iterrows():
    targets_array[index][int(row['Target'])] = 1


# put targets into dataframe so target and data get shuffled in sync
data['t0'] = targets_array[:,0]
data['t1'] = targets_array[:,1]
data['t2'] = targets_array[:,2]
data['t3'] = targets_array[:,3]
data['t4'] = targets_array[:,4]
data['t5'] = targets_array[:,5]
data['t6'] = targets_array[:,6]
data['t7'] = targets_array[:,7]
data['t8'] = targets_array[:,8]
data['t9'] = targets_array[:,9]
data['t10'] = targets_array[:,10]
data['t11'] = targets_array[:,11]
data['t12'] = targets_array[:,12]
data['t13'] = targets_array[:,13]
data['t14'] = targets_array[:,14]
data['t15'] = targets_array[:,15]
data['t16'] = targets_array[:,16]



# shuffle data
data = data.sample(frac=1.)


# split into test and train sets
n_test = len(data) // 10
n_train = len(data) - n_test

train = data[0:n_train]
test = data[n_train:-1]


# get list of features to pull out relevant columns for training
#print(data.columns.values)
#print(data)





features =  ['DII 170', 'DII 171', 'DII 172', 'DII 173', 'DII 174', 'DII 175',
 'DII 176', 'DII 177', 'DII 178', 'DII 179', 'DIII 180', 'DIII 181', 'DIII 182',
 'DIII 183', 'DIII 184', 'DIII 185', 'DIII 186', 'DIII 187', 'DIII 188',
 'DIII 189', 'AVR 190', 'AVR 191', 'AVR 192', 'AVR 193', 'AVR 194', 'AVR 195',
 'AVR 196', 'AVR 197', 'AVR 198', 'AVR 199', 'AVL 200', 'AVL 201', 'AVL 202',
 'AVL 203', 'AVL 204', 'AVL 205', 'AVL 206', 'AVL 207', 'AVL 208', 'AVL 209',
 'AVF 210', 'AVF 211', 'AVF 212', 'AVF 213', 'AVF 214', 'AVF 215', 'AVF 216',
 'AVF 217', 'AVF 218', 'AVF 219', 'V1 220', 'V1 221', 'V1 222', 'V1 223', 'V1 224',
 'V1 225', 'V1 226', 'V1 227', 'V1 228', 'V1 229', 'V2 230', 'V2 231', 'V2 232',
 'V2 233', 'V2 234', 'V2 235', 'V2 236', 'V2 237', 'V2 238', 'V2 239', 'V3 240',
 'V3 241', 'V3 242', 'V3 243', 'V3 244', 'V3 245', 'V3 246', 'V3 247', 'V3 248',
 'V3 249', 'V4 250', 'V4 251', 'V4 252', 'V4 253', 'V4 254', 'V4 255', 'V4 256',
 'V4 257', 'V4 258', 'V4 259', 'V5 260', 'V5 261', 'V5 262', 'V5 263', 'V5 264',
 'V5 265', 'V5 266', 'V5 267', 'V5 268', 'V5 269', 'V6 270', 'V6 271', 'V6 272',
 'V6 273', 'V6 274', 'V6 275', 'V6 276', 'V6 277', 'V6 278', 'V6 279']




train_x = train[features]
train_y = train[['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8','t9', 't10', 't11', 't12', 't13', 't14', 't15', 't16']]

test_x = test[features]
test_y = test[['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8','t9', 't10', 't11', 't12', 't13', 't14', 't15', 't16']]




# convert to numpy arrays for tensorflow
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

# get values for n_features, n_out
print(train_x.shape)
print(train_y.shape)


##############################################################################
# network
##############################################################################

lr = 0.1                        # learning rate for Gradient Descent
alpha = 0.01                    # alpha for regularization ( usually ~ 1/n_weights)

n_epochs = 200                   # how many times to loop entire dataset
n_out = train_y.shape[1]        # number of classes
n_features = train_x.shape[1]   # number of input features
n_hidden = 32                  # number of hidden nodes
n_convoluted = n_features // 2 - 1      # input data after 1st convolution


# used as placeholders in graph, x, y input data will fill these variables
x = tf.placeholder(tf.float32, shape=[n_features, 1], name='x')              # (1, n_features)
y = tf.placeholder(tf.float32, shape=[1, n_out], name='y')                   # (n_out, 1)





###############################################################################################
# weights that will be learned through training
##############################################################################################
          

# input layer
W1 = tf.Variable(tf.random_normal([n_convoluted, n_hidden]), name='W1')   
b1 = tf.Variable(tf.ones([n_hidden]), name='b1')                          


# residual layer
Wr = tf.Variable(tf.random_normal([n_features, n_hidden]), name='Wr')


# output layer
W2 = tf.Variable(tf.random_normal([n_hidden, n_out]), name='W2')            
b2 = tf.Variable(tf.ones([n_out], name='b2'))                               



##############################################################################################
# network layers
###############################################################################################

#----------------------------------------------------------------------------------------------
# convolution layer
k = tf.constant([.1, .8, .1], dtype=tf.float32, name='k')
kernel = tf.reshape(k, [int(k.shape[0]), 1, 1], name='kernel')
stride = 2
x_in    = tf.reshape(x, (1, n_features, 1), name='x_in')
convoluted = tf.squeeze(tf.nn.conv1d(x_in, kernel, stride, 'VALID'))
convoluted = tf.reshape(convoluted, (1, n_convoluted))
#----------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------
# first hidden layer
layer1 = tf.nn.sigmoid(tf.add(tf.matmul(convoluted, W1), b1))                    
#----------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------
# batch normalization
#----------------------------------------------------------------------------------------------
x_mean, x_var = tf.nn.moments(x, axes=[0])
normalized = tf.nn.batch_normalization(layer1, x_mean, x_var, offset=0, scale=1, variance_epsilon=0.001)
normalized = tf.expand_dims(normalized, 1)



#----------------------------------------------------------------------------------------------
# residual
#----------------------------------------------------------------------------------------------
residual = tf.add(tf.matmul(tf.transpose(x), Wr), normalized)        
residual = tf.reshape(residual, (1, n_hidden))



#----------------------------------------------------------------------------------------------
# output layer
# make predictions using trained network
prediction = tf.nn.sigmoid(tf.add(tf.matmul(residual, W2), b2))          
#----------------------------------------------------------------------------------------------





################################################################################################
# training equations
###############################################################################################
cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction) # how did we do?
regularization = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)                      # keep weights from blowing up
loss = cost + alpha * regularization                                           # used to calculate gradients

train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)             # update weights




############################################################################################
# tensorFlow Session - creates the graph that is the network
# and runs the network
############################################################################################
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

            x_batch = np.reshape(tx[i], (n_features, 1))
            y_batch = np.reshape(ty[i], (1, n_out))
            
            summary, cost_value, prediction_value = sess.run([train_op, cost, prediction], feed_dict={x: x_batch, y: y_batch})
            writer.add_summary(summary, i)
        
        #print("Weights", np.sum(sess.run(W1)), np.sum(sess.run(W2)))
        print("Epoch: : %i, cost %f" % (j, cost_value))

    
    #------------------  test  -----------------------------------------------------
    correct = 0
    print("Predicted, Actual")
    for k in range(n_test-1):

        x_batch = np.reshape(test_x[k], (n_features, 1))
        y_batch = np.reshape(test_y[k], (1,n_out))

        prediction_value = sess.run([prediction], feed_dict={x: x_batch, y: y_batch})
        print(np.argmax(prediction_value), np.argmax(y_batch))
        if np.argmax(prediction_value) == np.argmax(y_batch): correct += 1

    print("Correct %d / %d" % (correct, n_test))


