
# http://github.com/timestocome


# Attempt to build a Deep residual  network
# Simple example using the Iris Data set for testing

# Deep Residual Learning for Image Recognition ( MS Research )
# https://arxiv.org/pdf/1512.03385.pdf

# Batch Normalization
# https://arxiv.org/pdf/1502.03167.pdf



import numpy as np
import pandas as pd

from sklearn import datasets


import os
import sys
import timeit

import theano
import theano.tensor as T

theano.on_unused_input='ignore'


# setup theano
GPU = True
if GPU:
    print("Device set to GPU")
    try: theano.config.device = 'gpu'
    except: pass    # its already set
    theano.config.floatX = 'float32'
else:
    print("Running with CPU")



# tuning values -- no reason to keep passing constants into a function
learning_rate = 0.1
n_epochs = 4
batch_size = 2

L1_reg = 0.0
L2_reg = 0.0001

n_hidden = 6
n_inputs = 4            # 4 features
n_outputs = 3           # 3 types of iris


rng = np.random.RandomState(27)
  



#################################################################################
# load and prep dataset
###############################################################################
def load_data():

    data = datasets.load_iris()

    x_in = data.data         
    y_in = data.target         
    

    n_samples = len(x_in)

    # convert x to 0-1.
    max_x = np.max(x_in)
    x = x_in / max_x


    # convert y to one hot vectors
    y = np.zeros((n_samples, 3))
    for i in range(n_samples):
        y[i][y_in[i]] = 1.


    # shuffle
    xy = np.concatenate((x, y), axis=1)
    np.random.shuffle(xy)

    # resplit after shuffle
    x = xy[:, 0:4]
    y = xy[:, 4:7]

    return x, y


    
x_in, y_in = load_data()
n_samples = len(x_in)


n_test = n_samples // 10
n_valid = n_samples // 10
n_train = n_samples - n_valid - n_test 


# split into sets, oldest dates are training, testing next, validation most current
# and convert to numpy arrays
train_x = x_in[0:n_train]
test_x = x_in[n_train : n_train + n_test]
valid_x = x_in[n_train + n_test : len(x_in)]


train_y = y_in[0:n_train]
test_y = y_in[n_train : n_train + n_test]
valid_y = y_in[n_train + n_test : len(y_in)]




# load data into shared memory so it can be stored on gpu
def shared_dataset(data_x, data_y):

    # everything on the gpu is stored as floats 
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

    # we need ints for the targets so cast it back
    #return shared_x, T.cast(shared_y, 'int32')
    return shared_x, shared_y


test_x, test_y = shared_dataset(test_x, test_y)
valid_x, valid_y = shared_dataset(valid_x, valid_y)
train_x, train_y = shared_dataset(train_x, train_y)


# compute number of minibatches for training, validation and testing
n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_x.get_value(borrow=True).shape[0] // batch_size

  
print("number of samples: train, test, valid", n_train, n_test, n_valid)


# compute batch renormalization constants
input_mean = np.mean(x_in)
input_var = np.var(x_in)



def shuffle_idx():
    indexes = np.arange(0, n_train_batches)
    np.random.shuffle(indexes)
    return indexes

###################################################################################
# Hidden Forward Feed Layer
# input                     complete
# batch normalization       complete
# relu                      complete
####################################################################################
class HiddenLayer(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):

        # set up weights
        if W is None:
            W_values = np.asarray(rng.uniform(
                low = -np.sqrt(2./(n_in + n_out)),
                high = np.sqrt(2./(n_in + n_out)),
                size = (n_in, n_out)
            ), dtype=theano.config.floatX)

        self.W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out, ), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.params = [self.W, self.b]
        self.input = input

        # forward feed
        self.output = T.nnet.relu(T.dot(self.input, self.W) + self.b)


#####################################################################################
# Residual layer
# input + output from previous layer
######################################################################################
class ResLayer(object):

    def __init__(self, input, input_previous, n_in, n_in_p, W=None, b=None):

        # set up weights
        if W is None:
            W_values = np.asarray(rng.uniform(
                low = -np.sqrt(2./(n_in + n_in_p)),
                high = np.sqrt(2./(n_in + n_in_p)),
                size = (n_in, n_in_p)
            ), dtype=theano.config.floatX)

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.params = [self.W]
        self.input = input
        self.input_previous = input_previous

        # relu( F(x) + x )
        self.output = T.nnet.relu(T.dot(self.input, self.W) + self.input_previous)


#####################################################################################
# Batch normalization
# scale input data
######################################################################################
class BatchNormLayer(object):

    def __init__(self, input):

        # batch normalization  
        bn_mean_value = np.asarray(input_mean, dtype=theano.config.floatX)
        bn_variance_sqrt_value = np.asarray(np.sqrt(input_var), dtype=theano.config.floatX)

        x_mean = theano.shared(value=bn_mean_value, name='x_mean', borrow=True)
        x_sq_var = theano.shared(value=bn_variance_sqrt_value, name='x_sq_var', borrow=True)

        self.output = ( input - x_mean ) / x_sq_var


###################################################################################
# Output layer
# input + input from hidden layer 
# fully connected                   complete
# softmax                           complete
####################################################################################

class OutputLayer(object):

    def __init__(self, input, n_in, n_out):

        # set up fully connected weights
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out), dtype=theano.config.floatX), name='b', borrow=True)

        self.params = [self.W, self.b]

        self.input = input

        # compute probability of y given x
        # predict y given x - axis 1 is the column representing our output
        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


    
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(n_outputs-1), T.argmax(y[0])])
    

    def softmax(self, y):
        return -T.mean(self.p_y_given_x - y)


    # count the number of classes we missed on this mini-batch and return the mean
    def errors(self, y):
        return T.mean(T.neq(self.y_pred, T.argmax(y[0])))





########################################################################################
# Define network
########################################################################################

class Network(object):

    def __init__(self, input, n_in, n_hidden, n_out):

        # create layers
        self.bnLayer1 = BatchNormLayer(input=input)
        self.hiddenLayer = HiddenLayer(input=self.bnLayer1.output, n_in=n_inputs, n_out=n_hidden)
        self.resLayer = ResLayer(input=self.bnLayer1.output, input_previous=self.hiddenLayer.output, n_in=n_in, n_in_p=n_hidden)
        self.outputLayer = OutputLayer(input=self.resLayer.output, n_in=n_hidden, n_out=n_out)

        # regularization to prevent over training
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.outputLayer.W).sum() + abs(self.resLayer.W).sum()
        self.L2 = (self.hiddenLayer.W **2).sum() + (self.outputLayer.W **2).sum() + abs(self.resLayer.W).sum()

        # outputs
        self.negative_log_likelihood = self.outputLayer.negative_log_likelihood
        self.softmax = self.outputLayer.softmax 
        self.errors = self.outputLayer.errors
        
        # weights and biases to train
        self.params = self.hiddenLayer.params + self.outputLayer.params + self.resLayer.params

        # input data
        self.input = input

        # make predictions
        self.y_pred = self.outputLayer.y_pred

    



########################################################################################
# Define network
########################################################################################

def buildNetwork():

    # build the model
    index = T.lscalar()         # mini-batch index

    x = T.matrix('x')           # input data
    y = T.matrix('y')           # target labels


    # set up theano functions
    classifier = Network(x, n_in=n_inputs, n_out=n_outputs, n_hidden=n_hidden)

    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2


    test_model = theano.function(inputs=[index], outputs=classifier.errors(y), 
                givens={ x: test_x[index * batch_size:(index+1) * batch_size],
                         y: test_y[index * batch_size:(index+1) * batch_size]
                })

    validate_model = theano.function(inputs=[index], outputs=classifier.errors(y),
                givens={ x: valid_x[index * batch_size:(index+1) * batch_size],
                         y: valid_y[index * batch_size:(index+1) * batch_size] 
                })

    # get derivatives and apply to weights and bias
    d_params = [T.grad(cost, param) for param in classifier.params]
    updates = [ (param, param - learning_rate * d_param) for param, d_param in zip(classifier.params, d_params)]

    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
            givens={ x: train_x[index * batch_size:(index+1) * batch_size],
                     y: train_y[index * batch_size:(index+1) * batch_size]
            })



########################################################################################
# Train network
########################################################################################



    
    # train the model
    validation_frequency = 50      # how often to test validation examples
    best_validation_loss = np.inf   # best score on validation examples
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 0

    while epoch < n_epochs:

        epoch += 1

        # shuffle training data for each epoch
        indexes = shuffle_idx()

        for minibatch_index in range(n_train_batches):
            idx = minibatch_index
            minibatch_avg_cost = train_model(indexes[idx])

            if minibatch_index % validation_frequency == 0:

                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i, validation accurracy %f %%' % (epoch, minibatch_index, 100. - this_validation_loss * 100.))
                    

                # if best run so far?
                if this_validation_loss < best_validation_loss:
                    best_validation_loss = this_validation_loss

                    # try test ( hold out data )
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print("*****     Best score on hold out data %f %%" % (100. - test_score * 100.) )
                    
    end_time = timeit.default_timer()


    print("Optimization complete ")
    print("Best validation loss ", 100. - best_validation_loss * 100.)
    print("Best hold out loss ", 100. - test_score * 100.)
    print("Run time ", (end_time - start_time))


    return classifier

    

    
#################################################################################################
# run code to build and train network
################################################################################################
classifier = buildNetwork()


#################################################################################################
# make predictions using trained classifier
################################################################################################
print('***********************************************************')
print('Check model predictions')
def predict(classifier):

    # compile a predictor function
    predict_model = theano.function( inputs=[classifier.input], outputs=classifier.y_pred, allow_input_downcast=True)

    x = x_in.astype('float32')
    y = y_in.astype('float32')

    predicted_values = predict_model(x)

    total_predictions = np.unique(predicted_values, return_counts=True)
    print(total_predictions)

    '''
    print("Check predictions against true values:")
    for i in range(len(x)):
        predict = [0, 0, 0]
        predict[predicted_values[i]] = 1
        print(predict, y[i])
    '''


predict(classifier)