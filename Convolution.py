#!/python


import numpy as np
import pylab
from PIL import Image

import struct
from array import array as pyarray


import theano
from theano import tensor as T
from theano.tensor.nnet import conv

# seed random numbers
rng = np.random.RandomState(27)

# create 4D tensor for input
input = T.tensor4(name='input')

# initalize shared weights
w_shp = (2, 3, 9, 9)
w_bound = np.sqrt(3 * 9 * 9)
W = theano.shared(np.asarray(rng.uniform(
                                low = - 1.0 / w_bound,
                                high = 1.0 / w_bound,
                                size = w_shp),
                                dtype = input.dtype), name = 'W')
                                
# initalize shared bias with random numbers instead of zeros
# to simulate learning for this example, normally these are 
# learned as well
b_shp = (2,)
b = theano.shared(np.asarray(rng.uniform( low = -.5, high = .5, size = b_shp),
                                dtype = input.dtype), name = 'b')
                                
# build convolution filters
conv_out = conv.conv2d(input, W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# function to compute filtered images
filter = theano.function([input], output)



# open image for processing
image = Image.open('3wolfmoon.png')
image_data = np.asarray(image, dtype='float64') / 256

# convert image to 4D tensor of shape (1, 3, height, width) ? 3 is RGB
# convert image from (h, w, c) to (1, color, h, w)
img_ = image_data.transpose(2, 0, 1).reshape(1, 3, 639, 516)

# run image through the filters we made
filtered_img = filter(img_)

# plot original image and first and second components of output
pylab.subplot(1, 3, 1)
pylab.axis('off')
pylab.imshow(image)
pylab.gray()

# filtered image is a minibatch of size 1, index is 0
pylab.subplot(1, 3, 2)
pylab.axis('off')
pylab.imshow(filtered_img[0, 0, :, :])

pylab.subplot(1, 3, 3)
pylab.axis('off')
pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()

