
# http://github.com/timestocome
# convert example rnn-rbm from Lisa Labs that learns and generates music
# to do so with text from Alice in Wonderland



# started with this code:
# Author: Nicolas Boulanger-Lewandowski
# University of Montreal (2012)
# RNN-RBM deep learning tutorial
# More information at http://deeplearning.net/tutorial/rnnrbm.html


#import glob
import os
import sys

import pickle

import numpy as np
import pylab

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams



# set up 
np.random.seed(27)
rng = RandomStreams(seed=np.random.randint(1 << 30))




# setup theano
GPU = True
if GPU:
    print("Device set to GPU")
    try: theano.config.device = 'gpu'
    except: pass    # its already set
    theano.config.floatX = 'float32'
else:
    print("Running with CPU")



# constants
n_chain = 15        # gibbs chain length
n_rnn = 64          # rnn nodes
n_rbm = 64          # rbm nodes
n_epochs = 100       # times to loop over data
     
#####################################################################################
# load data 
# chars are 1-49 inclusive
# data is broken into phrases, shortest is 2 chars, longest is 319 chars
#####################################################################################

# read in char and word dictionarys and set up functions to convert ints to words
working_dir = os.getcwd()
data_directory = os.getcwd() + '/char_sentences'
os.chdir(data_directory)            # move to data file dir


file_list = os.listdir(os.getcwd()) # list data files
file_list.pop(0)                    # remove (%$#^% .DS_Store

n_samples = len(file_list)
batch_size = 50                      # highest and lowest possible values ( # of chars )   
                                     # 49 uniques but we start at 1 so add 1     

# read in files
x = []
for f in file_list:
    file_in = np.asarray(np.load(f), dtype=theano.config.floatX)
    x.append(file_in)



# convert each input vector into a matrix of one hot vectors
dataset = []
for i in x:                         # read in a phrase 
    phrase = []
    for j in i:                     # for each char in phrase
        z = np.zeros(batch_size)
        z[j] = 1.
        phrase.append(z)
    dataset.append(phrase)


# return to our project home dir
os.chdir(working_dir)
print("data loaded", len(x))


######  utilities #############
# convert ints back into chars
char_dict = pickle.load(open('char_index_dictionary.pkl', 'rb'))


def chars_to_string(chars):

    translation = [char_dict.get(k) for k in chars]
    
    text = []
    for i in translation:
        if i == None:
            text.append('_')
        else:
            text.append(i[0])

    text = np.asarray(text).flatten()
    print( text.astype('|S1').tostring().decode('utf-8') )




##########################################################################################
# RBM
##########################################################################################

# Construct a k-step Gibbs chain starting at v for an RBM.
# 
# k length of Gibbs chain
def build_rbm(v, W, bv, bh, k=n_chain):
  
    # obtain an approximate sequence from a probability distribution
    def gibbs_step(v):

        # hidden nodes
        mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h, dtype=theano.config.floatX)
        
        # visible/input nodes
        mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
        v = rng.binomial(size=mean_v.shape, n=1, p=mean_v, dtype=theano.config.floatX)

        return mean_v, v

    # create the sequence from the gibbs steps
    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v], n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]

    monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = monitor.sum() / v.shape[0]

    # cost 
    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()

    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates



#  Utility to initialize a matrix shared variable with normally distribution
def shared_normal(num_rows, num_cols, scale=1):
    return theano.shared(np.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))


#  Utility to initialize a vector shared variable with zero elements
def shared_zeros(*shape):
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX))



# visible is possible chars (50 in this case), n_hidden is rbm, n_hidden_recurrent is rnn
def build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent):
   

    W = shared_normal(n_visible, n_hidden, 0.01)
    bv = shared_zeros(n_visible)
    bh = shared_zeros(n_hidden)
    Wuh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
    Wuv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
    Wvu = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
    Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bu = shared_zeros(n_hidden_recurrent)

    params = W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu  # learned parameters as shared variables

    v = T.matrix()                              # a training sequence
    u0 = T.zeros((n_hidden_recurrent,))         # initial value for the RNN hidden units


    # calculate values for hidden and visible nodes
    def recurrence(v_t, u_tm1):
        bv_t = bv + T.dot(u_tm1, Wuv)
        bh_t = bh + T.dot(u_tm1, Wuh)
        generate = v_t is None
        if generate:
            v_t, _, _, updates = build_rbm(T.zeros((n_visible,)), W, bv_t, bh_t, k=25)

        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))
        return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]


    # loop over recurrence 
    (u_t, bv_t, bh_t), updates_train = theano.scan(
        lambda v_t, u_tm1, *_: recurrence(v_t, u_tm1),
        sequences=v, outputs_info=[u0, None, None], non_sequences=params)

    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:], k=15)

    updates_train.update(updates_rbm)

    # symbolic loop for sequence generation
    (v_t, u_t), updates_generate = theano.scan(
        lambda u_tm1, *_: recurrence(None, u_tm1),
        outputs_info=[None, u0], non_sequences=params, n_steps=200)

    return (v, v_sample, cost, monitor, params, updates_train, v_t, updates_generate)



# Simple class to train an RNN-RBM and generate sample sequences.
class RnnRbm:

    # n_hidden RBM hidden units
    # n_hidden_recurrent RNN hidden units
    # Is r lowest value (0, 50) 50 uniques in input data
    def __init__( self, n_hidden=n_rbm, n_hidden_recurrent=n_rnn, lr=0.001, r=(0, 50) ):
        
        self.r = r

        (v, v_sample, cost, monitor, params, updates_train, v_t, updates_generate) = build_rnnrbm( r[1] - r[0], n_hidden, n_hidden_recurrent)

        gradient = T.grad(cost, params, consider_constant=[v_sample])
        updates_train.update( ((p, p - lr * g) for p, g in zip(params, gradient)))
        self.train_function = theano.function( [v], monitor, updates=updates_train)
        self.generate_function = theano.function( [], v_t, updates=updates_generate)



    # loop over full data set once per epic
    def train(self, input, batch_size=batch_size, num_epochs=n_epochs):
       
        dataset = input
       
        try:
            for epoch in range(num_epochs):
                np.random.shuffle(dataset)
                costs = []

                for s, sequence in enumerate(dataset):                  # for each input vector 

                    for i in range(0, len(sequence), batch_size):       # 0...sequence length, batch_size = step
                        cost = self.train_function(sequence[i:i + batch_size])
                        costs.append(cost)

                print('Epoch %i/%i Cost %.2f' % (epoch + 1, num_epochs, np.mean(costs)))
                self.generate()
                print("*********************************************************************")
                sys.stdout.flush()

        except KeyboardInterrupt:
            print('Interrupted by user.')




    def generate(self):

        sample = np.array(self.generate_function().astype(int))
        
        # convert sample to ints
        ints = []
        for i in sample:
            ints.append(i.argmax())
              
        # convert sample to chars
        text = chars_to_string(ints)
    
        return text

       



def test_rnnrbm(batch_size=batch_size, num_epochs=n_epochs):

    model = RnnRbm()
    model.train(dataset, batch_size=batch_size, num_epochs=num_epochs)
    return model


##########################################################################
# train network 
################################################################
model = test_rnnrbm()


