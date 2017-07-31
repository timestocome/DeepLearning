# http://github.com/timestocome


# I stripped this down to use for another project.
# It's a nice clear example if you're just beginning.

# The Iris data set is here:
# http://archive.ics.uci.edu/ml/datasets/Iris


import pickle

import numpy as np
import pandas as pd

import os
import sys
import timeit


import theano
import theano.tensor as T

# setup theano
GPU = True
if GPU:
    print("Device set to GPU")
    try: theano.config.device = 'gpu'
    except: pass    # its already set
    theano.config.floatX = 'float32'
else:
    print("Running with CPU")

rng = np.random.RandomState(27)

#####################################################################################
# network setup
#####################################################################################

n_epochs = 10
batch_size = 1          #

learning_rate = 0.01
L1_reg = 0.0
L2_reg = 0.0001

n_hidden = 5
n_features = 4          # 4 input features
n_classes = 3           # possible classes 3 types of iris



 
####################################################################################
# load in data
####################################################################################

# read in data
data = pd.read_csv('Iris.csv')
data.columns = ['f1','f2','f3','f4','target']

# shuffle data
data = data.sample(frac=1)



# convert target from string to int
data['t0'] = 0

data.loc[data['target'] == 'Iris-virginica', 't0'] = 0
data.loc[data['target'] == 'Iris-setosa', 't0'] = 1
data.loc[data['target'] == 'Iris-versicolor', 't0'] = 2

# drop string column
data = data.drop('target', 1)



# split in to features and targets and train, test, validation sets
n_data = len(data)
n_test = 13
n_valid = 13
n_train = n_data - (n_test + n_valid)


data_in = data[['f1', 'f2', 'f3', 'f4']]
data_out = data[['t0']]

data_train_x = data_in[0:n_train]
data_test_x = data_in[n_train:n_train + n_test]
data_valid_x = data_in[n_train + n_test: -1]

data_train_y = data_out[0:n_train]
data_test_y = data_out[n_train:n_train + n_test]
data_valid_y = data_out[n_train + n_test : -1]



# load data into shared memory so it can be stored on gpu
def shared_dataset(data_x, data_y):

    # everything on the gpu is stored as floats 
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

    # we need ints for the targets so cast it back
    return shared_x, T.cast(shared_y, 'int32')

test_x, test_y = shared_dataset(data_test_x, data_test_y)
valid_x, valid_y = shared_dataset(data_valid_x, data_valid_y)
train_x, train_y = shared_dataset(data_train_x, data_train_y)


# compute number of minibatches for training, validation and testing
n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_x.get_value(borrow=True).shape[0] // batch_size



###################################################################################
# Hidden Layer
###################################################################################
class HiddenLayer(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):

        self.input = input

        if W is None:
            W_values = np.asarray(rng.uniform( low = -0.2, high = 0.2, size = (n_in, n_out)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out, ), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.output = T.tanh(T.dot(self.input, self.W) + self.b)

        self.params = [self.W, self.b]


####################################################################################
# Logistic Regression Layer
####################################################################################

class LogisiticRegression(object):

    def __init__(self, input, n_in, n_out):

        self.input = input

        # init weights
        self.lr_W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='lr_W', borrow=True)
        self.lr_b = theano.shared(value=np.zeros((n_out), dtype=theano.config.floatX), name='lr_b', borrow=True)

        # compute error
        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.lr_W) + self.lr_b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.lr_W, self.lr_b]

    # compute cost
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    # count the number of classes we missed on this mini-batch and return the mean
    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y.T))





########################################################################################
# Create MLP network
########################################################################################

class MLP(object):

    def __init__(self, input, n_in, n_hidden, n_out):

        # input data
        self.input = input


        # create layers
        self.hiddenLayer = HiddenLayer(input=self.input, n_in=n_in, n_out=n_hidden)
        self.logisticRegressionLayer = LogisiticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)

        # regularization to prevent over training
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logisticRegressionLayer.lr_W).sum()
        self.L2 = (self.hiddenLayer.W **2).sum() + (self.logisticRegressionLayer.lr_W **2).sum()

        # outputs
        self.negative_log_likelihood = self.logisticRegressionLayer.negative_log_likelihood
        self.errors = self.logisticRegressionLayer.errors
        
        # weights and biases to train
        self.params = self.hiddenLayer.params + self.logisticRegressionLayer.params


       

########################################################################################
# Stochastic gradient descent optimization
########################################################################################

def train_model():

    # build the model
    index = T.lscalar()             # mini-batch index
    x = T.matrix('x')               # input features
    y = T.imatrix('y')              # target labels

    classifier = MLP(x, n_in=n_features, n_out=n_classes, n_hidden=n_hidden)


    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2

    # get derivatives and apply to weights and bias
    d_params = [T.grad(cost, param) for param in classifier.params]
    updates = [ (param, param - learning_rate * d_param) for param, d_param in zip(classifier.params, d_params)]


    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
            givens={ x: train_x[index * batch_size:(index+1) * batch_size],
                     y: train_y[index * batch_size:(index+1) * batch_size]
            })


    test_model = theano.function(inputs=[index], outputs=classifier.errors(y), 
                givens={ x: test_x[index * batch_size:(index+1) * batch_size],
                         y: test_y[index * batch_size:(index+1) * batch_size]
                })


    validate_model = theano.function(inputs=[index], outputs=classifier.errors(y),
                givens={ x: valid_x[index * batch_size:(index+1) * batch_size],
                         y: valid_y[index * batch_size:(index+1) * batch_size] 
                })



    # train the model
    validation_frequency = 10                   # how often to test validation examples
    best_validation_loss = np.inf               # best score on validation examples
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 0
    target_score = 0.055
    target_hit = False

    while epoch < n_epochs and target_hit==False:       # run until target accuracy is hit or max runs

        epoch += 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)

            if epoch % validation_frequency == 0:

                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print('epoch %i, minibatch %i, validation error %f %%' % (epoch, minibatch_index, this_validation_loss * 100.))
                    
                # if best run so far?
                if this_validation_loss < best_validation_loss:
                    best_validation_loss = this_validation_loss

                    # try test ( hold out data )
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print("Best error on hold out data %f %%" % (test_score * 100.) )

                    if test_score < target_score:
                        # save model
                        with open('best_model.pkl', 'wb') as f:
                            pickle.dump(classifier, f)
                            target_hit  = True
                        break
                

    end_time = timeit.default_timer()

    print("Optimization complete ")
    print("Best validation loss ", best_validation_loss * 100.)
    print("Best hold out loss ", test_score * 100.)
    print("Run time ", (end_time - start_time))


    # print weight and bias data
    for p in classifier.params:
        print("----------------------")
        print(p)
        print(p.eval())
        print(p.shape.eval())
        



################################################################################
# run
################################################################################    
train_model()