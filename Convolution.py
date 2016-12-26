# http://github.com/timestocome

# adapted from https://github.com/lisa-lab/DeepLearningTutorials

import numpy as np

import theano
from theano import tensor as T 
from theano.tensor.nnet import conv

import pylab
from PIL import Image



#  init
rng = np.random.RandomState(27)
input = T.tensor4(name='input')
height = width = 512                # image size

w_shape = (2, 3, 9, 9)              # filters, image depth, filter_height, filter_width
w_bound = np.sqrt(3 * 9 * 9)        # depth, height, width

# randomly init weights should act as an edge detector
W = theano.shared(np.asarray(rng.uniform(
    low = -1./w_bound,
    high = 1./w_bound,
    size = w_shape), dtype=input.dtype), name = 'W')

# 2 filter maps so two bias weights are required
b_shape = (2,)
b = theano.shared(np.asarray(rng.uniform(
    low = -.5, high = .5, size=b_shape),
    dtype = input.dtype), name = 'b')


# convolvute the image
conv_out = conv.conv2d(input, W)

# dimshuffle reshapes the tensor 
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# create funtions
f = theano.function([input], output)

# open an image 
image = Image.open('kitten.jpg')
image = np.asarray(image, dtype='float64') / 256

# convert to 4d array
image_ = image.transpose(2, 0, 1).reshape(1, 3, height, width)
filtered_image = f(image_)

# plot original, first and second outputs
pylab.subplot(1, 3, 1)
pylab.axis('off')
pylab.imshow(image)
pylab.gray()

pylab.subplot(1, 3, 2)
pylab.axis('off')
pylab.imshow(filtered_image[0, 0, :, :]) 

pylab.subplot(1, 3, 3) 
pylab.axis('off')
pylab.imshow(filtered_image[0, 1, :, :]) 
pylab.show()