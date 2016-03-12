#!/python



# adapted from original source code
# http://deeplearning.net/tutorial/

# useful papers
# https://www.cs.swarthmore.edu/~meeden/cs81/s08/DahlLaTouche.pdf



import os, sys, timeit
import numpy as np
import gzip, pickle

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams


##################################################################################
# if possible run on GPU rather than CPU
##################################################################################

GPU = True
if GPU:
    print ("Trying to run under a GPU. ")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print ("Running with a CPU. ")
     
     
     
     
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
 
##########################################################################################
# Logistic Regression Layer
# activation = softmax(dot(x, w) + b)
##########################################################################################
class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
    
        # initialize parameters 
        # shared variables keep their state between iterations
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
                        
    
###################################################################################################################
# Restricted Boltzmann Machine
####################################################################################################################

# Restricted Boltzmann Machine
class RBM(object):


    def __init__(self, input=None, n_visible=784, n_hidden=500, W=None, hbias=None, vbias=None, numpy_rng=None, theano_rng=None):
        
        # init
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # init random number generator
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(42)
            
        # we can save and restore our position in the random number generator    
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # init weights if none loaded
        if W is None:
         
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low = -4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high = 4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size = (n_visible, n_hidden)
                ), dtype=theano.config.floatX )
                
            # for gpu processing
            W = theano.shared(value=initial_W, name='W', borrow=True)

        # create shared variable for hidden units bias
        if hbias is None:
            hbias = theano.shared(value=np.zeros(n_hidden, dtype = theano.config.floatX), name = 'hbias', borrow = True)
        
        # create visible bias shared variable
        if vbias is None:
            vbias = theano.shared( value = np.zeros( n_visible, dtype = theano.config.floatX ), name='vbias', borrow=True )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        
        # shared variables for this class
        self.params = [self.W, self.hbias, self.vbias]
        
    # used to compute gradient
    def free_energy(self, v_sample):

        wx_b = T.dot(v_sample, self.W) + self.hbias             # W * x + hb
        vbias_term = T.dot(v_sample, self.vbias)                # v * vb
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)     # Sum
        
        return -hidden_term - vbias_term                            


    # propagate visible unit activations to hidden units
    def propup(self, vis):
    
        # Note that we return also the pre-sigmoid activation of the layer. 
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]


    # compute hidden unit activations given a sample of visibles
    def sample_h_given_v(self, v0_sample):
        
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean, dtype=theano.config.floatX)
                                             
        return [pre_sigmoid_h1, h1_mean, h1_sample]
        
        
        
    # propagate hidden units activation to visible units
    def propdown(self, hid):
        
        # Note that we return also the pre_sigmoid_activation of the layer.
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]



    # compute the activation of the visible given the hidden sample
    def sample_v_given_h(self, h0_sample):

        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean, dtype=theano.config.floatX)
        
        return [pre_sigmoid_v1, v1_mean, v1_sample]


    # gibbs sampling from hidden state
    def gibbs_hvh(self, h0_sample):
       
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
       
        return [pre_sigmoid_v1, v1_mean, v1_sample,  pre_sigmoid_h1, h1_mean, h1_sample]


    # gibbs sampling starting from visible units
    def gibbs_vhv(self, v0_sample):
     
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
     
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]


    # one step of Contrastive Divergence or Persistent Contrastive Divergence
    # Contrastive re-inits chain for each input image, 
    # CD does not wait for chain to converge 
    # Persistent just updates the chain 
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        
        # persistent: None for CD. For PCD, 
        # k: number of Gibbs steps to do in CD-k/PCD-k

        
        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent


        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        ###########################################################
        # scan performs a loop along an input sequence and produces an 
        # output at every time step
        # scan can see the previous K time steps
        # scan can replace for loops ( which don't exist in Theano )
        (
            [   pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )


        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]
        cost = T.mean(self.free_energy(self.input)) - T.mean( self.free_energy(chain_end))
        
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        
        
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast( lr, dtype=theano.config.floatX )
            
            
        # pseudo-likelihood is a better proxy for PCD    
        if persistent:
            updates[persistent] = nh_samples[-1]                         # persistent must be a shared variable
            monitoring_cost = self.get_pseudo_likelihood_cost(updates) 
              
        # reconstruction cross-entropy is a better proxy for CD
        else:   
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

        return monitoring_cost, updates
       

    # Stochastic approximation to the pseudo-likelihood
    def get_pseudo_likelihood_cost(self, updates):
        
        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    # Approximation to the reconstruction error
    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        
        # Note that this function requires the pre-sigmoid activation as
        # input. We need this optimization for the
        # cross-entropy since sigmoid of numbers larger than 30. (or
        # even less then that) turn to 1. and numbers smaller than
        # -30. turn to 0 which in terms will force theano to compute
        # log(0) and therefore we will get either -inf or NaN as
        # cost.

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)), axis=1 ))

        return cross_entropy
          
     

###########################################################################################
# Deep Belief Network
##########################################################################################
class DBN(object):
    
    """
    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784, hidden_layers_sizes=[500, 500], n_outs=10):
        """This class is made to support a variable number of layers.

        n_ins: dimension of the input to the DBN
        hidden_layers_sizes: intermediate layers size, must contain at least one value
        n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels


        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
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

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)



    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions for performing one step of
        gradient descent at a given layer.

        :param train_set_x: Shared var. that contains all datapoints used for training the RBM
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k
        '''

        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use
      

        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # using CD-k here (persisent=None) for training each RBM.
            cost, updates = rbm.get_cost_updates(learning_rate, persistent=None, k=k)

            
            fn = theano.function(
                inputs=[index, learning_rate], 
                outputs=cost, 
                updates=updates,
                # http://stackoverflow.com/questions/35622784/what-is-the-right-way-to-pass-inputs-parameters-to-a-theano-function
                # inputs=[index, theano.In(learning_rate, value=0.1)], outputs=cost, updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end] })


            
            pretrain_fns.append(fn)

        return pretrain_fns



    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        # generates theano functions for training and testing

         # datasets: It is a list that contain all the datasets;
         #               the has to contain three pairs, `train`,
         #               `valid`, `test` in this order, where each pair
         #               is formed of two Theano variables, one for the
         #               datapoints, the other for the labels
         # batch_size: size of a minibatch
         # learning_rate: learning rate used during finetune stage

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        #n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
        #n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))


        train_fn = theano.function(inputs=[index], outputs=self.finetune_cost, updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]})


        test_score_i = theano.function( [index], self.errors,
            givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]})


        valid_score_i = theano.function( [index], self.errors,
            givens={
                self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size: (index + 1) * batch_size ]})


        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]


        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score

# pretraining = 100 training_epochs = 1000 put it at 10 for testing
def test_DBN(finetune_lr=0.1, pretraining_epochs=100, pretrain_lr=0.01, k=1, training_epochs=1000, dataset='mnist.pkl.gz', batch_size=10):
    """
    Demonstrates how to train and test a Deep Belief Network.

    This is demonstrated on MNIST.

    finetune_lr: learning rate used in the finetune stage
    pretraining_epochs: number of epoch to do pretraining
    pretrain_lr: learning rate to be used during pre-training
    k: number of Gibbs steps in CD/PCD
    training_epochs: maximal number of iterations ot run the optimizer
    dataset: path the the pickled dataset
    batch_size: the size of a minibatch
    """

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # numpy random generator
    numpy_rng = np.random.RandomState(123)
    
    print ('... building the model')
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=28 * 28, hidden_layers_sizes=[1000, 1000, 1000], n_outs=10)

    
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print ('... getting the pretraining functions')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size, k=k)

    print ('... pre-training the model')
    start_time = timeit.default_timer()

    for i in range(dbn.n_layers):

        for epoch in range(pretraining_epochs):

            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index, lr=pretrain_lr))
            print ('Pre-training layer %i, epoch %d, cost ' % (i, epoch), np.mean(c))

    end_time = timeit.default_timer()

    print ('The pretraining code for file ran for %.2fm' % ((end_time - start_time) / 60.))
    
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print ('... getting the finetuning functions')
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
        datasets=datasets, batch_size=batch_size, learning_rate=finetune_lr)


    print ('... finetuning the model')
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    
    improvement_threshold = 0.995  
    validation_frequency = min(n_train_batches, patience / 2)
                                 
    best_validation_loss = np.inf
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
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % 
                    ( epoch, minibatch_index + 1, n_train_batches, 100.0 - this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches, 100.0 - test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete with best validation score of %f %%, obtained at iteration %i with test performance %f %%') 
            % (100.0 - best_validation_loss * 100., best_iter + 1, 100.0 - test_score * 100.))
    print ('The fine tuning code for file ran for %.2fm' % ((end_time - start_time) / 60.))



####################################################################################
# Run network
####################################################################################
test_DBN()
