#!/python

from __future__ import print_function

import os
import sys
import timeit
import gzip, pickle

import numpy as np
import random

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d, sigmoid
from theano.tensor.signal import downsample


GPU = True
if GPU:
    print ("Trying to run under a GPU. ")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print ("Running with a CPU. ")
        
        

#########################################
# to do 
# ? add drop out
# original leNet5 has two hidden layers - add one in?

# meh - best kaggle score was .94971 with this net



#################################################################################################
# LeNet5 http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
# input 32 * 32
# conv maps 6 @ 28x28
# pool 2,2
# conv maps 16 @ 10x10
# pool 2,2
# hidden 120
# hidden 84
# output 10
#################################################################################################

#################################################################################################
# parameters to tweak 

learning_rate = 0.1      # how fast does net converge - bounce out of local mins
                         # makes a noticeable difference in accuracy
                        
L2_reg = 0.001            # lambda - scaling factor for regularization term

n_epochs = 3            # max number of times we loop through full training set
batch_size = 100          # number of training examples per batch - smaller is slower but better accuracy( above 20)

n_kerns = [20, 40]       # number of kernels per layer [ 1st layer, 2nd layer]
n_hidden = 100           # hidden layer size


rng = np.random.RandomState(42)             # seed for random


def ReLU(z): return T.maximum(0.0, z)       # ReLU is supposed to be the best for image recognition networks
#activation_function = ReLU                  # T.tanh, T.nnet.sigmoid are also possible, T.tanh works better than sigmoid
activation_function = T.tanh


initial_weight_scale = 1.0                  # 1.0 for ReLU, 6. for tanh, 24. for sigmoid
#################################################################################################
# load up data 

# Kaggle subset of MNIST dataset, images are 28x28 images, 0.0-1.0 0 being blank, 1.0 darkest mark on image
# labels are single ints 0-9
# training set is 40,000 images and labels
# testing and validation sets are 1,000 images and labels each
# submission set is 28,000
##################################################################################################


# read in the data files and format as needed
import LeNetLoadData
train_set, valid_set, test_set, kaggle_set = LeNetLoadData.load_data_wrapper()


# shared variables can be quickly loaded onto the gpu
def shared_dataset(data_xy, borrow=True):
       
    data_x, data_y = zip(*data_xy)    
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    return shared_x, T.cast(shared_y, 'int32')


test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)
kaggle_set_x, kaggle_set_y = shared_dataset(kaggle_set)


    
# set up constants
# double front slash (//) divide and round down to floor
n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size         # 40,000 or 200,000
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size         #  1,000
n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size           #  1,000
n_kaggle_batches = kaggle_set_x.get_value(borrow=True).shape[0] // batch_size       # 28,000


##########################################################################################

              
##########################################################################################
# Hidden Layer 
# uses tanh instead of sigmoid
# activation = tanh(dot(x,W) + b)
##########################################################################################         
class HiddenLayer(object):
    # rng = random state
    # input = output from previous layer
    # n_in = number of inputs, length of previous layer 
    # n_out = number of hidden units
    # activation can be T.tanh or T.nnet.sigmoid
    # W = weights
    # b = bias
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=activation_function):
        
        self.input = input
        
        # initial weights sqrt( +/-6. / (n_in + n_hidden)), multiply by 4 for sigmoid
        if W is None:
            W_values = np.asarray(rng.uniform(
                                    low = -np.sqrt(initial_weight_scale/(n_in + n_out)),
                                    high = np.sqrt(initial_weight_scale/(n_in + n_out)),
                                    size = (n_in, n_out)
                                    ), dtype = theano.config.floatX)
            W = theano.shared(value = W_values, name = 'W', borrow = True)
            
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)
        
        self.W = W
        self.b = b
        
        # calculate linear output using dot product + b, else use tanh or sigmoid or ReLU
        lin_output = T.dot(input, self.W) + self.b
        self.output = activation_function(lin_output)
        
        # update weights and bias
        self.params = [self.W, self.b]
        
    def regularization(self):
        return (L1_reg * abs(self.W).sum()) + (L2_reg * abs(self.W **2).sum())     
   
     
##########################################################################################
# Logistic Regression Layer
# activation = softmax(dot(x, w) + b)
##########################################################################################
class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
    
        # initialize parameters 
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
   
        # map between 0-1 so each output is a probability - makes training easier
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
   
        # compute predicted class - index of largest value
        self.y_predict = T.argmax(self.p_y_given_x, axis=1)
   
        self.params = [self.W, self.b]
        self.input = input
   
        

    # using mean instead of sum allows for different batch sizes with out adjusting learning rate
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    

    # number of errors in the mini batch
    def errors(self, y):
        return T.mean(T.neq(self.y_predict, y))
    
   
   
   
##########################################################################################
# LeNetConvPoolLayer
# Convolutional layers downsize the inputs using filters
# Convolution multiplies each filter number by the incoming pixel then divides by sum of filter numbers
# A pooling layer is needed between each convolutional layer
##########################################################################################         
class LeNetConvPoolLayer(object):
    
    def __init__(self, rng, input, filter_shape, image_shape, pool_size=(2, 2)):
        
            assert image_shape[1] == filter_shape[1]
            self.input = input

            # there are "num input feature maps * filter height * filter width inputs to each hidden unit
            fan_in = np.prod(filter_shape[1:])
            
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /  pooling size
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(pool_size))
           
            # initialize weights with random weights
            W_bound = np.sqrt(initial_weight_scale / (fan_in + fan_out))
            self.W = theano.shared( np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                        dtype=theano.config.floatX), borrow=True)
            


           
            # setup bias - one per feature map
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
            
            # convolve input feature maps with filters
            conv_out = conv2d(
                input = input, 
                filters = self.W,
                filter_shape = filter_shape,
                image_shape = image_shape)
                
            # downsample each feature map using maxpooling
            pooled_out = downsample.max_pool_2d(
                input = conv_out,
                ds = pool_size,
                ignore_border = True
            )
            
            # add bias
            self.output = activation_function(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            
            # store weights and biases for this layer
            self.params = [self.W, self.b]
            
            # track input
            self.input = input


############################################################################################
# create network
def evaluate_lenet5():
    
    index = T.lscalar()                 # index to minibatch
    x = T.matrix('x')                   # input images
    y = T.ivector('y')                  # correct labels
    
    #############################
    # build model
    ############################
    print( "Building the model...........")
    
    # reshape input images to match conv pool layer
    layer0_input = x.reshape((batch_size, 1, 28, 28))
    
    # construct first conv pooling layer
    # filter output size = 28 - 5 + 1
    # max pooling size = 24/2
    # so output = (batch_size, n_kerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(rng, input = layer0_input, 
                                image_shape = (batch_size, 1, 28, 28), 
                                filter_shape = (n_kerns[0], 1, 5, 5), 
                                pool_size = (2,2))
    
    # construct second layer
    # filter output size = 12 - 5 + 1
    # pooling output size = 8 / 2 
    # so output = (batch_size, n_kerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(rng, input = layer0.output, 
                                image_shape = (batch_size, n_kerns[0], 12, 12),
                                filter_shape = (n_kerns[1], n_kerns[0], 5, 5),
                                pool_size = (2, 2))
                                
    # fully connected hidden layer (batch_size, num_pixels) -- (500, 800)
    layer2_input = layer1.output.flatten(2)
    
    # fully connected sigmoid layer
    layer2 = HiddenLayer (rng, input = layer2_input, 
                            n_in = n_kerns[1] * 4 * 4,
                            n_out = n_hidden,
                            activation = activation_function )
                            
    # Softmax output layer
    layer3 = LogisticRegression(input = layer2.output, n_in = n_hidden, n_out = 10)
    
    
    # regularization
    regularization =  0.5 * L2_reg * sum([(layer3.W**2).sum()]) 
    
    # negative_log_likelihood cost
    cost = layer3.negative_log_likelihood(y) + regularization
    
    # compute errors
    test_model = theano.function( [index], layer3.errors(y), 
                givens={ x : test_set_x[index * batch_size: (index + 1) * batch_size],
                         y : test_set_y[index * batch_size: (index + 1) * batch_size]})
                         
    validate_model = theano.function ( [index], layer3.errors(y),
                givens={ x : valid_set_x[index * batch_size: (index + 1) * batch_size],
                         y : valid_set_y[index * batch_size: (index + 1) * batch_size]})
    
    kaggle_predictions = theano.function( [index], layer3.y_predict, 
                 givens={ x : kaggle_set_x[index  * batch_size: (index + 1) * batch_size]})
              
                         
    # weights and biases to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params
    
    # gradients list for all layers
    grads = T.grad(cost, params)
    
    # update model weights and biases
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]
    
    train_model = theano.function([index], cost, updates = updates, 
                                givens={ x : train_set_x[index * batch_size: (index + 1) * batch_size],
                                         y : train_set_y[index * batch_size: (index + 1) * batch_size]})
         
                                         
         
                                         
    
    #############################
    # train model
    ############################
    print("training......")
    
    
    # early stopping to prevent over fitting
    patience = 10000                # min number of examples to view
    patience_increase = 2           # wait this long before updating best
    improvement_threshold = 0.995   # min improvement to consider
    validation_frequency = min(n_train_batches, patience // 2)                   # how often to check validation set

    
    best_validation_loss = np.inf
    best_epoch = 0
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs): # and (not done_looping):   let's let it run a bit longer
        epoch = epoch + 1
        
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if iter % 100 == 0:
                print ('_______________________training iteration______________________________ ', iter)
                
            cost_ij = train_model(minibatch_index)
                 
            # compute zero one loss on validation set
            validation_losses = [validate_model(i) for i in range(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
            #print('epoch %i, minibatch %i/%i, validation %f %%' %(epoch, minibatch_index + 1, n_train_batches, 100.0 - this_validation_loss * 100.))
                
            # if best score to date
            if this_validation_loss < best_validation_loss:
                
                if this_validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                # update best scores
                best_validation_loss = this_validation_loss
                best_epoch = epoch
                    
                # test on test set
                test_losses = [test_model(i) for i in range(n_test_batches)]
                test_score = np.mean(test_losses)
                print(('***** epoch %i, minibatch %i/%i, test score of best model %f %%') % (epoch, minibatch_index + 1, n_train_batches, 100.0 - test_score * 100.))
                    
                    
                   
                 
                    
                if patience <= iter:
                    done_looping = True
                    break
                    
    end_time = timeit.default_timer()
    
    print("***************************************************************************************")
    print("........................Optimization complete..........................................")
    print("Best validation score of %f %% obtained at iteration %i with test score of %f %%" %
                        (100.0 - best_validation_loss * 100., best_epoch, 100.0 - test_score * 100.))
    print("Total runtime: %.2f " % ((end_time - start_time)/60.))
    print("........................................................................................")
    
    
    ######  print to text file as well ##############
    run_file = open('LeNet5_lastRun.txt', 'w')
    run_file.write("Best validation score of %f %% obtained at iteration %i with test score of %f %%" %
                        (best_validation_loss * 100., best_epoch, test_score * 100.))
    run_file.write("Total runtime: %.2f " % ((end_time - start_time)/60.))
    run_file.close()
    
    
    
    ################################
    # validate model
    ################################
    # run through Kaggle data and print estimates to submission file
    # runs through the Kaggle validation data and writes out a submission file  
    test_results = []  
    batch_results = [kaggle_predictions(j) for j in range(n_kaggle_batches)]
    test_results.append(batch_results)
    test_array = np.asarray(test_results)

    test_array = test_array.flatten()
    
    
   # save kaggle submission to disk
    np.savetxt('kaggle_leNet_results.csv', 
                        np.c_[range(1,len(test_array)+1),test_array], 
                        delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
                 
    
    
    #########  save weights, biases etc ? ###############################################
    with open('lenet_best_model.pkl', 'wb') as f: pickle.dump(params, f)
    

############################################################################################
# run network
evaluate_lenet5()
  

    
    