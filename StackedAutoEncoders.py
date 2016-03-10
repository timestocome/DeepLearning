#!/python

# adapted from original source code
# http://deeplearning.net/tutorial/

# papers
# http://www.iro.umontreal.ca/~vincentp/Publications/denoising_autoencoders_tr1316.pdf
# http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf


# appears to be up and running, didn't feel like running it for hours to test results


from __future__ import print_function

import os, sys, timeit
import numpy as np
import gzip, pickle

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


#####################################################################################
# load data 
#####################################################################################
# pickled, zipped data file 
filename = 'mnist.pkl.gz'
    
# Load the dataset
with gzip.open(filename, 'rb') as f:
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set = pickle.load(f)
        
        
        
# shared variables can be quickly loaded onto the gpu
def shared_dataset(data_xy, borrow=True):
       
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    return shared_x, T.cast(shared_y, 'int32')


test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)

datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

              
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
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        
        self.input = input
        
        # initial weights sqrt( +/-6. / (n_in + n_hidden)), multiply by 4 for sigmoid
        if W is None:
            W_values = np.asarray(rng.uniform(
                                    low = -np.sqrt(6./(n_in + n_out)),
                                    high = np.sqrt(6. /(n_in + n_out)),
                                    size = (n_in, n_out)
                                    ), dtype = theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4.
            W = theano.shared(value = W_values, name = 'W', borrow = True)
            
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)
        
        self.W = W
        self.b = b
        
        # calculate linear output using dot product + b, else use tanh or sigmoid
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        
        # update weights and bias
        self.params = [self.W, self.b]
        
        
        
##################################################################################################
# Denoising Autoencoder
# Attempts to reconstruct images
#################################################################################################

class dA(object):

   
    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500, W=None, bhid=None, bvis=None):
    
            self.n_visible = n_visible
            self.n_hidden = n_hidden
            
            if not theano_rng:
                theano_rng = RandomStreams(numpy_rng.randint( 2 ** 30))
                
            if not W:
                initial_W = np.asarray(numpy_rng.uniform(
                                        low = -4 * np.sqrt(6. / (n_hidden + n_visible)),
                                        high = 4 * np.sqrt(6. / (n_hidden + n_visible)),
                                        size = ( n_visible, n_hidden)                                        
                ), dtype = theano.config.floatX )
                
                W = theano.shared(value = initial_W, name='W', borrow=True)
                
            if not bvis:
                bvis = theano.shared( value = np.zeros(n_visible, dtype = theano.config.floatX), borrow = True)
            
            if not bhid:
                bhid = theano.shared( value = np.zeros(n_hidden, dtype = theano.config.floatX), name = 'b', borrow = True)
                
            self.W = W
            self.b = bhid
            self.b_prime = bvis
            self.W_prime = self.W.T     # matching weights W, W-transpose
            self.theano_rng = theano_rng
            
            if input == None:            # if no input, generate some
                self.x = T.dmatrix(name='input')
            else:
                self.x = input
                
            self.params = [self.W, self.b, self.b_prime]
            
            
    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size = input.shape, n = 1, p = 1-corruption_level, dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        
    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        
    # computer error cost and weight updates for one step    
    def get_cost_updates(self, corruption_level, learning_rate):
    
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1-z), axis=1) # vector containing cost function
        cost = T.mean(L)    
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
        
        return (cost, updates)
        
        
  
##########################################################################################
# Logistic Regression Layer
# activation = softmax(dot(x, w) + b)
##########################################################################################
class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
    
        # initialize parameters 
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
   
        # map input to hyperplane to determine output
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
   
        # compute predicted class
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
# The Stacked Encoder Network
##########################################################################################

class SdA(object):
    
    # n_ins = inputs to this layer
    # n_layers_sizes = list of ints containing at least one intermediate layer size
    # n_out = output classes for the network
    # corruption_levels = list of corruption levels for each of the hidden layers 

    def __init__ (self, numpy_rng, theano_rng=None, n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10, corruption_levels=[0.1, 0.1] ):
        

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            
        self.x = T.matrix('x')          # input image
        self.y = T.ivector('y')         # output classes

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP



        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]


            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
                                        
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this layer
            dA_layer = dA(numpy_rng = numpy_rng,
                          theano_rng = theano_rng,
                          input = layer_input,
                          n_visible = input_size,
                          n_hidden = hidden_layers_sizes[i],
                          W = sigmoid_layer.W,
                          bhid = sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
            
            
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)


        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        
        self.errors = self.logLayer.errors(self.y)



   # Generates a list of functions, each of them implementing one
   #     step in training the dA corresponding to the layer with same index.
   #     The function will require as input the minibatch index, and to train
   #     a dA you just need to iterate, calling the corresponding function on
   #     all minibatch indexes.
    def pretraining_functions(self, train_set_x, batch_size):
    
       
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch

        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
       
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size


        pretrain_fns = []
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(corruption_level, learning_rate)
            fn = theano.function(
                inputs=[
                    index,
                    corruption_level, 
                    learning_rate],
                    # http://stackoverflow.com/questions/35622784/what-is-the-right-way-to-pass-inputs-parameters-to-a-theano-function
                    #index, theano.In(corruption_level, value=0.2),     
                    #theano.In(learning_rate, value=0.1)], 
                    outputs=cost, updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end] })
            pretrain_fns.append(fn)

        return pretrain_fns



    # load up datasets and finetune things
    def build_finetune_functions(self, datasets, batch_size, learning_rate):
      

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        gparams = T.grad(self.finetune_cost, self.params)

        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams) ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]},
            name='train'
        )

        test_score_i = theano.function([index], self.errors,
            givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]},
            name='test'
        )

        valid_score_i = theano.function([index], self.errors,
            givens={
                self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]},
            name='valid'
        )

        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score



#     Demonstrates how to train and test a stochastic denoising autoencoder. This is demonstrated on MNIST.
def test_SdA(finetune_lr=0.1, pretraining_epochs=15, pretrain_lr=0.001, training_epochs=1000, dataset='mnist.pkl.gz', batch_size=1):

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
   
    numpy_rng = np.random.RandomState(42)
    print ('... building the model')
    
    # construct the stacked denoising autoencoder class
    sda = SdA(numpy_rng=numpy_rng, n_ins=28 * 28, hidden_layers_sizes=[1000, 1000, 1000], n_outs=10)


    #########################
    # PRETRAINING THE MODEL #
    #########################
    print ('... pre-training model')
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size)
    start_time = timeit.default_timer()
    
    
    ## Pre-train layer-wise
    corruption_levels = [.1, .2, .3]
    for i in range(sda.n_layers):
    
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
        
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print ('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c)))

    end_time = timeit.default_timer()

    print ('The pretraining ran for %.2fm' % ((end_time - start_time) / 60.))
    
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetunning the model')
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print((
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, with test performance %f %%')
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print(('The training code for file ran for %.2fm' % ((end_time - start_time) / 60.)),)


    
    
    
test_SdA()
