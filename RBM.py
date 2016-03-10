#!/python


# adapted from original source code
# http://deeplearning.net/tutorial/


# http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
# http://deeplearning4j.org/restrictedboltzmannmachine.html


"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""

from __future__ import print_function

import timeit
import gzip, pickle


try:
    import PIL.Image as Image
except ImportError:
    import Image


import numpy as np

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams






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
    
     
####################################################################################################
# image plotting functions
###################################################################################################
# scale to between 0-1
def scale_to_unit_interval(ndar, eps=1e-8):
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

# convert array to image matrix
def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array
     
     
     
###################################################################################################################
# Restricted Boltzmann Machine
####################################################################################################################

# Restricted Boltzmann Machine
class RBM(object):


    def __init__(self, input=None, n_visible=784, n_hidden=500, W=None, hbias=None, vbias=None, numpy_rng=None, theano_rng=None):
        

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # init random number generator
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(42)

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
        

    def free_energy(self, v_sample):

        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        
        return -hidden_term - vbias_term

    # propagate visible unit activations to hidden units
    def propup(self, vis):
    
        # Note that we return also the pre-sigmoid activation of the
        # layer. As it will turn out later, due to how Theano deals with
        # optimizations, this symbolic variable will be needed to write
        # down a more stable computational graph (see details in the
        # reconstruction cost function)
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


    # one step of CD-k or PCD-k
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        
        # persistent: None for CD. For PCD, 
        # k: number of Gibbs steps to do in CD-k/PCD-k

        # Returns a proxy for the cost and the updates dictionary. The
        # dictionary contains the update rules for weights and biases but
        # also an update of the shared variable used to store the persistent
        # chain, if one is used.

       
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
        # input.  To understand why this is so you need to understand a
        # bit about how Theano works. Whenever you compile a Theano
        # function, the computational graph that you pass as input gets
        # optimized for speed and stability.  This is done by changing
        # several parts of the subgraphs with others.  One such
        # optimization expresses terms of the form log(sigmoid(x)) in
        # terms of softplus.  We need this optimization for the
        # cross-entropy since sigmoid of numbers larger than 30. (or
        # even less then that) turn to 1. and numbers smaller than
        # -30. turn to 0 which in terms will force theano to compute
        # log(0) and therefore we will get either -inf or NaN as
        # cost. If the value is expressed in terms of softplus we do not
        # get this undesirable behaviour. This optimization usually
        # works fine, but here we have a special case. The sigmoid is
        # applied inside the scan op, while the log is
        # outside. Therefore Theano will only see log(scan(..)) instead
        # of log(sigmoid(..)) and will not apply the wanted
        # optimization. We can not go and replace the sigmoid in scan
        # with something else also, because this only needs to be done
        # on the last step. Therefore the easiest and more efficient way
        # is to get also the pre-sigmoid activation as an output of
        # scan, and apply both the log and sigmoid outside scan such
        # that Theano can catch and optimize the expression.


        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)), axis=1 ))

        return cross_entropy


# Train and test 
def test_rbm(learning_rate=0.1, training_epochs=15,
             dataset='mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=500):
    
    # n_chains: number of parallel Gibbs chains to be used for sampling
    # n_samples: number of samples to plot for each chain

    

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()       # index to a [mini]batch
    x = T.matrix('x')         # the data is presented as rasterized images

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden layer of chain)
    persistent_chain = theano.shared(np.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True)

    # construct the RBM class
    rbm = RBM(input=x, n_visible=28 * 28, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate, persistent=persistent_chain, k=15)


    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)


    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function( [index], cost, updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }, name='train_rbm' )

    plotting_time = 0.
    start_time = timeit.default_timer()

    for epoch in range(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print('Training epoch %d, cost is ' % epoch, np.mean(mean_cost))

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))
    

    #################################
    #     Sampling from the RBM     #
    #################################
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared( np.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX ))
            
    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    # add to updates the shared variable that takes care of our persistent chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = np.zeros( (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')
    
    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print(' ... plotting sample %d' % idx)
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save('samples.png')
    os.chdir('../')


#########################################################################################################
# run the rbm
#########################################################################################################
test_rbm()
