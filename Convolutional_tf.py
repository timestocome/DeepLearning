
# http://github.com/timestocome



import pickle
import numpy as np
import tensorflow as tf
import sys


# made some minor improvements, cleaned up code

# started with simple example from Machine Learning with TF book
# https://www.manning.com/books/machine-learning-with-tensorflow




# dataset from https://www.cs.toronto.edu/~kriz/cifar-10- python.tar.gz




#######################################################################
# load data
######################################################################

def unpickle(file):

    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()

    return dict




def clean(data):

    # n_samples, 3 colors, width, height
    imgs = data.reshape(data.shape[0], 3, 32, 32)
    
    # convert to greyscale
    grayscale_imgs = imgs.mean(1)

    # crop edges
    # n_samples, width = 24, height = 24
    cropped_imgs = grayscale_imgs[:, 4:28, 4:28]
    
    # reshape n_images, image data as 1d array
    img_data = cropped_imgs.reshape(data.shape[0], -1)
    img_size = np.shape(img_data)[1]
    
    # get avg value for each image
    means = np.mean(img_data, axis=1)
    meansT = means.reshape(len(means), 1)
    
    # get std for each image
    stds = np.std(img_data, axis=1)
    stdsT = stds.reshape(len(stds), 1)

    # normalize image data around mean
    adj_stds = np.maximum(stdsT, 1.0 / np.sqrt(img_size))
    normalized = (img_data - meansT) / adj_stds

    return normalized


def read_data(directory):
    
    names = unpickle('{}/batches.meta'.format(directory))['label_names'] 
    print('names', names)

    data, labels = [], [] 
    for i in range(1, 6):
        filename = '{}/data_batch_{}'.format(directory, i) 
        batch_data = unpickle(filename)

        if len(data) > 0:
            data = np.vstack((data, batch_data['data']))
            labels = np.hstack((labels, batch_data['labels'])) 
        else:
            data = batch_data['data'] 
            labels = batch_data['labels']

    print(np.shape(data), np.shape(labels))
    data = clean(data)
    data = data.astype(np.float32) 
    
    return names, data, labels


names, data, labels = read_data('./cifar-10-batches-py')

print(data.shape)





######################################################################
# network
#####################################################################
n_layer1_filters = 64
n_layer2_filters = 32
filter1_size = 5
filter2_size = 4
image_width = 24
image_height = 24
n_hidden = 512


n_batches = 200
n_epochs = 10



# inputs and outputs
x = tf.placeholder(tf.float32, [None, image_height * image_width])     # 24x24 images
y = tf.placeholder(tf.float32, [None, len(names)])  # number of names = number of categories

# 64 - 5x5 convolutional filters applied to input
W1 = tf.Variable(tf.random_normal([filter1_size, filter1_size, 1, n_layer1_filters]))
b1 = tf.Variable(tf.random_normal([n_layer1_filters]))

# 64, 5x5 convolution filters applied to layer 1 output
W2 = tf.Variable(tf.random_normal([filter2_size, filter2_size, n_layer1_filters, n_layer2_filters]))
b2 = tf.Variable(tf.random_normal([n_layer2_filters]))

# fully connected layer
W3 = tf.Variable(tf.random_normal([6 * 6 * n_layer2_filters, n_hidden]))
b3 = tf.Variable(tf.random_normal([n_hidden]))

# fully connected output layer
W_out = tf.Variable(tf.random_normal([n_hidden, len(names)]))
b_out = tf.Variable(tf.random_normal([len(names)]))



def conv_layer(x, W, b):

    conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    conv_with_b = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(conv_with_b)
    
    return conv_out


def maxpool_layer(conv, k=2):
    return tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')



def model():

    x_reshaped = tf.reshape(x, shape=[-1, image_height, image_width, 1])

    # perform 1st convolution on input data
    conv_out1 = conv_layer(x_reshaped, W1, b1) 
    maxpool_out1 = maxpool_layer(conv_out1)
    norm1 = tf.nn.lrn(maxpool_out1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # perform 2nd convolution on 1st layer output
    conv_out2 = conv_layer(norm1, W2, b2)
    # local response normalization 
    # https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
    norm2 = tf.nn.lrn(conv_out2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75) 
    maxpool_out2 = maxpool_layer(norm2)

    # run through the fully connect and the output layers
    maxpool_reshaped = tf.reshape(maxpool_out2, [-1, W3.get_shape().as_list()[0]]) 
    local = tf.add(tf.matmul(maxpool_reshaped, W3), b3)
    local_out = tf.nn.relu(local)
    out = tf.add(tf.matmul(local_out, W_out), b_out) 

    return out


model_op = model()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_op, labels=y) )
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) 

correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

########################################################################################
# train
#######################################################################################


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # convert output to one hot labels for network
    onehot_labels = tf.one_hot(labels, len(names), on_value=1., off_value=0., axis=-1) 
    onehot_vals = sess.run(onehot_labels)

    batch_size = len(data) // n_batches
    print('batch size', batch_size)
    sys.stdout.flush()


    for j in range(n_epochs): 

        for i in range(0, len(data), batch_size): 
            batch_data = data[i:i+batch_size, :]
            batch_onehot_vals = onehot_vals[i:i+batch_size, :]
            _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: batch_data, y:batch_onehot_vals})
     
            

        print('Epoch: ', j, accuracy_val * 100. )
        sys.stdout.flush() # force printing
