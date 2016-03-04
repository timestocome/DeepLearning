#!/python


# built in
import numpy as np
import pickle, gzip
import timeit
import sys, os


# 3rd party
import theano 
import theano.tensor as T


################################################################################################
# single hidden layer netword - input, hidden, output
# input data is transformed to be linearly separable using hidden layer
# activations are non-linear using either tanh or sigmoid
# final layer is a softmax layer
# regularization done with L1 and L2 - L2 is just L1 squared
# this example uses tanh instead of sigmoid in hidden layer - supposedly allows for faster training
# training is done with stochastic gradient descent backpropagated through the net 
# all layers are fully connected
# ? use 4x sized weights for sigmoid than for tanh


#################################################################################################
# parameters used to tweak learning
##################################################################################################
learning_rate = 0.01    # how fast does net converge - bounce out of local mins
L1_reg = 0.00           # lambda - scaling factor for regularizations
L2_reg = 0.0001
n_epochs = 1000         # number of times we loop through full training set
batch_size = 20         # number of training examples per batch - smaller is slower but better accuracy 
n_hidden = 500          # number of nodes in hidden layer


#################################################################################################
# load up data 
# MNIST dataset, images are 28x28 images, 0.0-1.0 0 being blank, 1.0 darkest mark on image
# labels are single ints 0-9
# training set is 50,000 images and labels
# testing and validation sets are 10,000 images and labels each
##################################################################################################
 
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
   
# set up datasets
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]
    
    
# set up constants
# double front slash (//) divide and round down to floor
n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
              
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
        
        
   
     
         
##########################################################################################
# MultiLayer Perceptron Class
# Forward feed, backpropagation network with one or more hidden layers using non-linear activations
# output layer uses softmax
# input layer is the data
##########################################################################################                
class MLP(object):
    
    # random state (initializer)
    # input - input data
    # n_in - number of input units (28*28 for mnist)
    # n_hidden - number of hidden units in hidden layer
    # n_out - number of output labels (10 digits for mnist)
    def __init__(self, rng, input, n_in, n_hidden, n_out):
            
            # hidden layer, takes in weighted inputs and feeds them to the 
            # logistic regression layer
            self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)
            
            # logistic regression layer gets weighted hidden activations as input
            # and outputs the label (class) we think the image belongs to
            self.logisticRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)
                   
            # regularization
            self.L1 = ( abs(self.hiddenLayer.W).sum() + abs(self.logisticRegressionLayer.W).sum() )
            self.L2 = ( (self.hiddenLayer.W ** 2).sum() + (self.logisticRegressionLayer.W ** 2).sum() )
            
            # measure error
            self.negative_log_likelihood = ( self.logisticRegressionLayer.negative_log_likelihood )
            self.errors = self.logisticRegressionLayer.errors
            
            # parameters
            self.params = self.hiddenLayer.params + self.logisticRegressionLayer.params
            
            # data in
            self.input = input
            
     
     
##########################################################################################
# Logistic Regression 
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
    
    
    def sgd_optimization_mnist():    
    
        # build the model
        print("building model...")
    
        # setup variables
        index = T.lscalar()     # index to a mini batch
        x = T.matrix('x')       # image data
        y = T.ivector('y')      # labels
    

        classifier = LogisticRegression(input=x, n_inputs=28*28, n_outputs=10)
    
        cost = classifier.negative_log_likelihood(y)
        
         # by the model on a minibatch
        test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )

    
                                    
        # compute gradient of cost with respect to theta = (W, b)
        g_W = T.grad(cost=cost, wrt=classifier.W)
        g_b = T.grad(cost=cost, wrt=classifier.b)
    
        # update weights and biases
        updates = [(classifier.W, classifier.W - learning_rate * g_W),
                (classifier.b, classifier.b - learning_rate * g_b)]
                
        # training model description
        train_model = theano.function(inputs=[index], outputs=cost, updates=updates, 
                                givens={       x: train_set_x[index * batch_size:(index+1) * batch_size],
                                               y: train_set_y[index * batch_size:(index+1) * batch_size]
                                               })
                                               
        print("Training model.........")
        # initialize loop variables for stopping and checking progress
        patience = 5000                 # minimum number of examples to use
        patience_increase = 2           # wait at least this long before updtating best
        improvement_threshold = 0.995   # min significant improvement
        validation_frequency = min(n_train_batches, patience)
        best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()
        done_looping = False
        epoch = 0
    
        # for each training loop
        while ( epoch < n_epochs ) and ( not done_looping ):        
            epoch = epoch + 1
        
            # for each mini batch in the full data set
            for minibatch_index in range(n_train_batches):
                
                # train then move to next batch
                minibatch_avg_cost = train_model(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index
                
                # check progress
                if (iter + 1) % validation_frequency == 0:
                
                    validation_losses = [validate_model(i) for i in range(n_valid_batches)]

                    this_validation_loss = 1.0 - np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, current accuracy %f %%' % 
                            (epoch, minibatch_index+1, n_train_batches, this_validation_loss*100.))
                            
                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        
                        best_validation_loss = this_validation_loss
                    
                        test_losses = [test_model(i) for i in range(n_test_batches)]
                        test_score = 1.0 - np.mean(test_losses)
                    
                        print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') % 
                                    (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
                                
                        with open('best_model.pkl', 'wb') as f:
                            pickle.dump(classifier, f)
            
                if patience <= iter:
                    done_looping = True
                    break
                
        end_time = timeit.default_timer()    
        print(('Optimization complete with best validation score of %f %%, with test performance %f %%') %
                    (best_validation_loss * 100., test_score * 100.))
                    
        print(('The code ran for %d epochs, with %f epochs/sec' % 
                    (epoch, 1. * epoch / (end_time - start_time))), file=sys.stderr)
                    
  
    
    
    
###########################################################################################
# run the network on the dataset
def test_mlp():

    ##################################################
    # build the model
    ##################################################
    print("building the model......")
    
    index = T.lscalar()                   # index to minibatch
    x = T.matrix('x')                   # data in
    y = T.ivector('y')                  # output classes/labels
    rng = np.random.RandomState(42)     # seed for random

    classifier = MLP( rng=rng, input=x, n_in=28*28, n_hidden=n_hidden, n_out=10)
    cost = ( classifier.negative_log_likelihood(y) + (L1_reg * classifier.L1) + (L2_reg * classifier.L2) )
    
    test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
            })

    validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            })
                    
    # compute gradient of cost                
    gradients = [T.grad(cost, param) for param in classifier.params]
    updates = [(param, param - learning_rate * g) for param, g in zip(classifier.params, gradients)]
    
    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                givens={
                    x: train_set_x[index * batch_size:(index + 1) * batch_size],
                    y: train_set_y[index * batch_size:(index + 1) * batch_size]
                })
                
    ##################################################
    # train the model
    ##################################################            
    print("training ....")
    
     # early-stopping parameters
    patience = 10000                    # look as this many examples regardless
    patience_increase = 2               # wait this much longer when a new best is found
    improvement_threshold = 0.995        # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
    
        epoch = epoch + 1
    
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
    

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
            
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = 1.0 - np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, accuracy %f %%' %
                    ( epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                
                    #improve patience if loss improvement is good enough
                    if ( this_validation_loss < best_validation_loss * improvement_threshold ):
                        patience = max(patience, iter * patience_increase)
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = 1.0 - np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, best accuracy %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
          
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

##########################################################################################
# run network
#LogisticRegression.sgd_optimization_mnist()
test_mlp()