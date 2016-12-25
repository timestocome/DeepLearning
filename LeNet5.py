
# http://github.com/timestocome


import pickle
import gzip


import numpy as np

import os
import sys
import timeit

import theano
import theano.tensor as T
from theano.tensor.signal import pool 
from theano.tensor.nnet import conv2d


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
n_epochs = 20
batch_size = 200
n_filters_1 = 20
n_filters_2 = 50

L1_reg = 0.0
L2_reg = 0.0001

n_hidden = 50
n_pixels = 28 * 28      # image size (784 pixels)
n_classes = 10          # possible classes (0-9)

rng = np.random.RandomState(27)
  

####################################################################################
# load in data
####################################################################################
# load file into memory
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()



# load data into shared memory so it can be stored on gpu
def shared_dataset(data_xy):

    data_x, data_y = data_xy

    # everything on the gpu is stored as floats 
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

    # we need ints for the targets so cast it back
    return shared_x, T.cast(shared_y, 'int32')


test_x, test_y = shared_dataset(test_set)
valid_x, valid_y = shared_dataset(valid_set)
train_x, train_y = shared_dataset(train_set)


# compute number of minibatches for training, validation and testing
n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_x.get_value(borrow=True).shape[0] // batch_size



####################################################################################
# Convolutional and Pooling Layer

class ConvolutionalPoolLayer(object):

    def __init__(self, input, filter_shape, image_shape, poolsize=(2,2)):
        # batch_size * filter_height * filter_width
        fan_in = np.prod(filter_shape[1:])

        # feature_maps * filter_height * filter_width / pooling_size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))

        W_bounds = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(rng.uniform(
                    low = -W_bounds, high = W_bounds, size=filter_shape),
                    dtype=theano.config.floatX), borrow=True)    
                    
        # one bias per feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d( input = input, filters = self.W, filter_shape = filter_shape, input_shape = image_shape)

        # downsample using max pooling 
        pooled_out = pool.pool_2d( input = conv_out, ds = poolsize, ignore_border = True)

        # add bias
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]
        self.input = input



###################################################################################
# Hidden Layer
# use sqrt(6/# of weights) for tanh
# use 4 * sqrt(6/# of weights) for sigmoid
# use sqrt(2/# of weights) for relu
####################################################################################
class HiddenLayer(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):

        self.input = input

        if W is None:
            W_values = np.asarray(rng.uniform(
                low = -np.sqrt(6./(n_in + n_out)),
                high = np.sqrt(6./(n_in + n_out)),
                size = (n_in, n_out)
            ), dtype=theano.config.floatX)

        W = theano.shared(value=W_values, name='W', borrow=True)
        self.W = W

        if b is None:
            b_values = np.zeros((n_out, ), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
        self.b = b

        # T.nnet.sigmoid
        # T.nnet.relu
        # T.tanh
        self.output = T.tanh(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]


###################################################################################
# Logistic Regression Layer
# linear output layer
####################################################################################

class LogisiticRegression(object):

    def __init__(self, input, n_in, n_out):

        # init weights and bias to zero 
        # shared loads them onto gpu
        # borrow means they get updated immediately
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out), dtype=theano.config.floatX), name='b', borrow=True)

        # the equations for this layer
        # compute probability of y given x
        # predict y given x - axis 1 is the column representing our output
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.input = input

    # y[0] is the number of examples (rows) in our mini-batch
    # columns are our output classes
    # using mean, could use sum/mini-batch-count
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    # count the number of classes we missed on this mini-batch and return the mean
    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))





########################################################################################
# Create Lenet5 Network
########################################################################################
def build_network():


    index = T.lscalar()         # mini-batch index

    x = T.matrix('x')           # input data
    y = T.ivector('y')          # target labels


    # reshape the input from 1d vector to 4d (number in batch, depth, width, height)
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # first  convolutional layer ( 20-1x5x5 filters)
    layer0 = ConvolutionalPoolLayer(input = layer0_input, 
                                image_shape=(batch_size, 1, 28, 28),
                                filter_shape=(n_filters_1, 1, 5, 5), 
                                poolsize=(2,2))

    # image reduced from 28 x 28 to 28 - 5 + 1 = (24 x 24)
    # then (24 x 24)/(2, 2) = 12, 12

    # second convolutional layer ( 50-1x4x4 filters)
    layer1 = ConvolutionalPoolLayer(input = layer0.output, 
                                image_shape = (batch_size, n_filters_1, 12, 12),
                                filter_shape = (n_filters_2, n_filters_1, 5, 5),
                                poolsize = (2, 2))

    # fully connected hidden layer
    layer_2_input = layer1.output.flatten(2) # batch size, pixels after convolutions
    layer2 = HiddenLayer ( input = layer_2_input, 
                        n_in = n_filters_2 * 4 * 4,
                        n_out = n_hidden )
                        
    #  logistic regression layer
    layer3 = LogisiticRegression(input=layer2.output,
                            n_in = n_hidden, 
                            n_out = n_classes)


    # functions
    cost = layer3.negative_log_likelihood(y)
    params = layer3.params + layer2.params + layer1.params + layer0.params
    grads = T.grad(cost, params)

    updates = [ (param_i, param_i - learning_rate * grad_i)
                for param_i, grad_i in zip(params, grads)]

    train_model = theano.function( [index], cost, updates = updates,
                    givens = { x: train_x[index * batch_size: (index+1) * batch_size],
                                y: train_y[index * batch_size: (index+1) * batch_size]
                    })


    test_model = theano.function([index], layer3.errors(y), 
                givens = { x: test_x[index * batch_size: (index + 1) * batch_size],
                            y: test_y[index * batch_size: (index + 1) * batch_size]
                })

    validate_model = theano.function([index], layer3.errors(y),
                    givens = { x: valid_x[index * batch_size: (index + 1) * batch_size],
                                y: valid_y[index * batch_size: (index + 1) * batch_size]
                    })



    # train the network
    validation_frequency = 10     # how often to test validation examples
    best_validation_loss = np.inf   # best score on validation examples
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 1
    target_score = 0.055
    target_hit = False

    while epoch < n_epochs and target_hit==False:

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
                        target_hit  = True
                        break
                

    end_time = timeit.default_timer()


    print("Optimization complete ")
    print("Best validation loss ", best_validation_loss * 100.)
    print("Best hold out loss ", test_score * 100.)
    print("Run time ", (end_time - start_time))



if __name__ == '__main__':
    build_network()