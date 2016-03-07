#!/python

# Denoising Autoencoder
# maps input to a hidden layer then out to a reconstructed representation
# inputs are corrupted, outputs uncorrupted

# http://www.iro.umontreal.ca/~vincentp/Publications/denoising_autoencoders_tr1316.pdf
# http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf

# original source code from http://deeplearning.net/tutorial/deeplearning.pdf

from __future__ import print_function

import os, sys, timeit

import gzip, pickle

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import PIL.Image as Image


##################################################################################################
# parameters to tweak
n_visible = 784             # input image 28 * 28
n_hidden = 500              # hidden layer nodes
learning_rate = 0.1         # needs to be high enough to avoid local mins, and low enough not to bounce
training_epochs = 15        # number of loops through full training set
batch_size = 20             # number of inputs between updating weights         

#################################################################################################





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

n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
 

#################################################################################################
# Tile raster images
# create an image from a set of weights
#
# Transform an array with one flattened image per row, into an array in
#    which images are reshaped and layed out like tiles on a floor.

#    This function is useful for visualizing datasets whose rows are images,
#    and also columns of matrices for transforming those rows
#    (such as the first layer of a neural net).
#################################################################################################
# Scales all values in the ndarray ndar to be between 0 and 1 
def scale_to_unit_interval(ndar, eps=1e-8):

    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar



#    X: a 2-D array in which every row is a flattened image. ( 2D array or tuple of 4 channels )
#    img_shape: the original shape of each image ( height, width )
#    tile_shape: the number of images to tile (rows, cols)
#    output_pixel_vals: if output should be pixel values (i.e. int8 values) or floats
#    scale_rows_to_unit_interval: if the values need to be scaled before being plotted to [0,1] or not
#    returns array suitable for viewing as an image. ( 2D array with same type as X)

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # output shape created from input
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct dtype
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

##################################################################################################
# Denoising Autoencoder
# Attempts to reconstruct images
#################################################################################################

class dA(object):

    def __init__(self, numpy_rng, theano_rng=None, input=None, W=None, bhid=None, bvis=None):
    
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
    def get_cost_updates(self, corruption_level):
    
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1-z), axis=1) # vector containing cost function
        cost = T.mean(L)    
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
        
        return (cost, updates)
        
                
# test on mnist dataset
def test_dA():

     index = T.lscalar()            # mini batch index
     x = T.matrix('x')              # present data as images
     
     
     #######################################
     # building the model with no corruption          
     #######################################
     
     rng = np.random.RandomState(27)
     theano_rng = RandomStreams(rng.randint( 2**30 ))
     
     da = dA(numpy_rng = rng, theano_rng = theano_rng, input = x)
     cost, updates = da.get_cost_updates(corruption_level=0.0)
     
     train_da = theano.function([index], cost, updates=updates, 
                        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})
                        
                        
     start_time = timeit.default_timer()
     
     #####################################
     # training
     #####################################
     for epoch in range(training_epochs):
        
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
            
        print("Training epoch: %d, cost " % epoch, np.mean(c))
        
    
     end_time = timeit.default_timer()
     training_time = end_time - start_time
    
    
     print("The no corruption training ran for %.2f" % (training_time/60.))
    
     image = Image.fromarray(
            tile_raster_images( X = da.W.get_value(borrow=True).T,
                                img_shape = (28, 28), tile_shape = (10, 10), tile_spacing = (1, 1)))
     image.save('filters_corruption_0.png')
    
    
    #######################################
     # building the model with 30%          
     #######################################
     
     rng = np.random.RandomState(27)
     theano_rng = RandomStreams(rng.randint( 2**30 ))
     
     da = dA(numpy_rng = rng, theano_rng = theano_rng, input = x)
     cost, updates = da.get_cost_updates(corruption_level=0.3)
     
     train_da = theano.function([index], cost, updates=updates, 
                        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})
                        
                        
     start_time = timeit.default_timer()
     
     #####################################
     # training
     #####################################
     for epoch in range(training_epochs):
        
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
            
        print("Training epoch: %d, cost " % epoch, np.mean(c))
        
    
     end_time = timeit.default_timer()
     training_time = end_time - start_time
    
    
     print("The 30% corruption training ran for", (training_time/60.))
    
     image = Image.fromarray(
            tile_raster_images( X = da.W.get_value(borrow=True).T,
                                img_shape = (28, 28), tile_shape = (10, 10), tile_spacing = (1, 1)))
     image.save('filters_corruption_30.png')
            
            
#######################################################################################################            
test_dA()