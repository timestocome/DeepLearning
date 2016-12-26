# http://github.com/timestocome

# adapted from the Lisa Labs Theano tutorial
# Denoising Autoencoder
# maps input to a hidden layer then out to a reconstructed representation
# inputs are corrupted, outputs uncorrupted

# http://www.iro.umontreal.ca/~vincentp/Publications/denoising_autoencoders_tr1316.pdf
# http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf

# original source code from http://deeplearning.net/tutorial/deeplearning.pdf



import os
import sys
import timeit
import gzip
import pickle
import numpy as np

import theano
import theano.tensor as T 
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

# setup theano
GPU = True
if GPU:
    print("Device set to GPU")
    try: theano.config.device = 'gpu'
    except: pass    # its already set
    theano.config.floatX = 'float32'
else:
    print("Running with CPU")



import PIL.Image as Image


rng = np.random.RandomState(27)
theano_rng = RandomStreams(rng.randint(2 **30))

######################################
# constants - network tweaks and settings
n_visible = 28 * 28             # size of input image height * width
n_hidden = 500                  # hidden nodes
batch_size = 20
learning_rate = 0.1
epochs = 25
output_folder = 'DA_plots'

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


#####################################################################################
# misc functions
######################################################################################


# scale to between 0-1
def scale_to_unit_interval(ndar, eps=1e-8):
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

# reshape arrays into 2d images and tile images
def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                        scale_rows_to_unit_interval=True, output_pixel_vals=True):

    
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
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
                # if channel is None, fill it with zeros of the correct dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it in the output
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


#####################################################################################
# Denoising auto encoder class
####################################################################################


class DA(object):

    def __init__(self, input=None, W=None, b_hidden=None, b_visible=None):

            if not W:
                W_values = np.asarray(rng.uniform(
                    low = -4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high = 4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size = (n_visible, n_hidden)), dtype = theano.config.floatX)

                W = theano.shared(value=W_values, name='W', borrow=True)
            self.W = W
            self.W_prime = self.W.T

            if not b_visible:
                b_visible = theano.shared( value = np.zeros(n_visible, 
                                            dtype=theano.config.floatX), borrow=True)
            self.b_prime = b_visible 

            if not b_hidden:
                b_hidden = theano.shared( value = np.zeros(n_hidden, 
                                            dtype=theano.config.floatX), name = 'b', borrow=True)
            self.b = b_hidden

            if input == None:
                self.x = T.dmatrix(name='input')
            else: self.x = input

            self.params = [self.W, self.b, self.b_prime]


    # randomly zero out corruption_level number of input pixels
    def get_corrupted_input(self, input, corruption_level):

        # array of 1s and 0s as mask for corruption
        return theano_rng.binomial( size = input.shape, 
                                        n = 1.,
                                        p = 1. - corruption_level,
                                        dtype=theano.config.floatX ) * input


    def get_hidden_values(self, input):

        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    
    def get_reconstructed_input(self, hidden):

        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)


    def get_cost_updates(self, corruption_level):

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        loss = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(loss)

        gradients = T.grad(cost, self.params)
        updates = [ (param, param - learning_rate * gradient)
                    for param, gradient in zip(self.params, gradients)]
        
        return ( cost, updates )




def test_DA():

    index = T.lscalar()
    x = T.matrix('x')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # build the model with uncorrupted data
    
    da = DA(input=x)
    cost, updates = da.get_cost_updates(corruption_level=0.)
    train_da = theano.function([index], cost, updates=updates, 
                    givens = { x: train_x[index * batch_size: (index + 1) * batch_size]})

    start_time = timeit.default_timer()

    # train the model
    for epoch in range(epochs):

        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print("training... epoch %d, cost %.2f" % (epoch, np.mean(c, dtype='float64')))

    
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    print("trained with no corruption %.2f" % (training_time/60.))

    image = Image.fromarray(tile_raster_images(X=da.W.get_value(borrow=True).T,
                                            img_shape=(28,28), tile_shape=(10,10),
                                            tile_spacing=(1,1)))
    image.save('filters_corruption_0.png')


    # build the model with 30% corruption 
    
    da = DA(input=x)
    cost, updates = da.get_cost_updates(corruption_level=0.2)
    train_da = theano.function([index], cost, updates=updates, 
                    givens = { x: train_x[index * batch_size: (index + 1) * batch_size]})

    start_time = timeit.default_timer()

    # train the model
    for epoch in range(epochs):

        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print("training... epoch %d, cost %.2f" % (epoch, np.mean(c, dtype='float64')))

    
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    print("trained with 20%% corruption %.2f" % (training_time/60.))


    image = Image.fromarray(tile_raster_images(X=da.W.get_value(borrow=True).T,
                                            img_shape=(28,28), tile_shape=(10,10),
                                            tile_spacing=(1,1)))
    image.save('filters_corruption_30.png')


if __name__ == '__main__':
    test_DA()