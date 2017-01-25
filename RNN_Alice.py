
# http://github.com/timestocome


# adapt to Alice in Wonderland/Throug the Looking Glass
# convert to python 3
# clean up code
# improve code
# fix several errors in code



# started with this code:
# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python

import numpy as np
import pickle
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import os
import random

import theano
import theano.tensor as T 



# set up 
rng = np.random.RandomState(27)

# setup theano
GPU = True
if GPU:
    print("Device set to GPU")
    try: theano.config.device = 'gpu'
    except: pass    # its already set
    theano.config.floatX = 'float32'
else:
    print("Running with CPU")




#######################################################################
# open data files created in ParseDataIntoSentences.py
#####################################################################
# array of sentences
sentences = np.load('sentences.npy')

# convert words to int and ints to words
idx2word = np.load('idx2word.npy')
word2idx = pickle.load(open('word2idx.pkl', "rb"))

# list of words only appearing once that we stripped out for training
unique_word_list = np.load('unique_word_list.npy')
n_unique_words = len(unique_word_list)

idx2word_list = list(idx2word)      # use list format to convert words to ints


# convert sentence text to ints
def convert_input(idx2word_list):
    
    idx_sentences = []
    for sentence in sentences:
        idx_s = []
        for w in sentence:
            idx_s.append(idx2word_list.index(w))
            s = np.asarray(idx_s).astype('int32')
        idx_sentences.append(s)

    return idx_sentences


input = convert_input(idx2word_list)


 # convert output arrays of ints back into text
def output_to_text(z):
    text = []
    for i in z:
        if idx2word[i] == 'unique_word':
            text.append(unique_word_list[random.randint(0, n_unique_words-1)])
        else:
            text.append( idx2word[i] )
    return text

######################################################################
# network constants
#######################################################################

n_dimension = 8                    # dimension of word embedding ( longest sentence in ParseDataIntoSentences.py)
                                     # reducing this reduces the window size we loop over

n_vocabulary = len(idx2word)         # number of unique words 4041
n_hidden = n_vocabulary // 8         # hidden layer size


learning_rate = 0.001
epochs = 5000
n_sentences = len(input)

########################################################################
# utilities
#######################################################################


########################################################################
# RNN
########################################################################

class RNN:

    def __init__(self, D, M, V):

        self.D = D          # dimensionality of word embedding
        self.M = M          # hidden layer size
        self.V = V          # vocabulary size



        # initial weights
        We_values = np.asarray(rng.uniform(
                low = -np.sqrt(2./(V + D)), 
                high = np.sqrt(2./(V + D)),
                size = (V, D)), dtype = theano.config.floatX )
        self.We = theano.shared(value=We_values, name='We', borrow=True)

        Wx_values = np.asarray(rng.uniform(
                low = -np.sqrt(2./(D + M)), high = np.sqrt(2./(D + M)),
                size = (D, M)), dtype = theano.config.floatX )
        self.Wx = theano.shared(value=Wx_values, name='Wx', borrow=True)

        Wh_values = np.asarray( rng.uniform(
                low = -np.sqrt(2./(M + M)), high = np.sqrt(2./(M + M)),
                size = (M, M)), dtype = theano.config.floatX ) 
        self.Wh = theano.shared(value=Wh_values, name='Wh', borrow=True)
            
        bh_values = np.zeros((M), dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_values, name='bh')

        h_values = np.zeros((M), dtype=theano.config.floatX)
        self.h = theano.shared(value=h_values, name='h')

        Wo_values =  np.asarray(rng.uniform(
                low = -np.sqrt(2./(M + V)), high = np.sqrt(2./(M + V)),
            size = (M, V)), dtype = theano.config.floatX )
        self.Wo = theano.shared(value=Wo_values, name='Wo', borrow=True)

        bo_values = np.zeros(V)
        self.bo = theano.shared(value=bo_values, name='bo', borrow=True)


        self.parameters = [self.We, self.Wx, self.Wh, self.bh, self.h, self.Wo, self.bo]


        def save_weights():
            np.savez("RNN_Alice_weights.npz", *[p.get_value() for p in self.parameters])
        self.save_weights = save_weights
   
        



    def fit(self, X):

        N = len(X) 
        D = self.D
        M = self.M
        V = self.V

        thX = T.ivector('X')
        Ei = self.We[thX]                   
        thY = T.ivector('Y')

        # input:
        # [START, w1, w2, ..., wn]
        # target:
        # [w1,    w2, w3, ..., END]


         # feed forward equations
         # no training with tanh. sigmoid
        def recurrence(x_t, h_t1):
            h_t = T.nnet.relu( T.dot(x_t, self.Wx) + T.dot(h_t1, self.Wh) + self.bh )
            y_t = T.nnet.softmax( T.dot(h_t, self.Wo) + self.bo )
            return h_t, y_t

        # loop over feed forward equations once for each bit in the sequence 
        # send previous hidden output back through and collect prediction
        [h, y_predicted], _ = theano.scan(
            fn = recurrence,
            outputs_info = [self.h, None],
            sequences = Ei,
            n_steps = Ei.shape[0],
        )

        # probability of x given y
        py_x = y_predicted[:, 0, :]
        prediction = T.argmax(py_x, axis=1) # fetch most likely prediction

        # cost functions for gradients and tracking progress
        cost = -T.mean( T.log(py_x[T.arange(thY.shape[0]), thY]))       # cross entropy
        gradients = T.grad(cost, self.parameters)                       # derivatives

        updates = [(p, p - learning_rate * g) for p, g in zip(self.parameters, gradients)]

       
        self.predict_op = theano.function(inputs=[thX], outputs=prediction)

        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates, allow_input_downcast=True
        )

        costs = []
        n_total_words = sum((len(sentence)+1) for sentence in X)
        

        for i in range(epochs):


            X = shuffle(X)
            n_correct = 0
            cost = 0
            
            for j in range(N):
               
                input_sequence = X[j][1:-2]         # remove START/END and last word
                output_sequence = X[j][2:-1]        # remove START/END and first word

               
                # we set 0 to start and 1 to end
                c, p = self.train_op(input_sequence, output_sequence)
                cost += c
                
              
                # print "j:", j, "c:", c/len(X[j]+1)                
                for pj, xj in zip(p, output_sequence):
                    if pj == xj:
                        n_correct += 1

            if i % 10 == 0:            
                print ("i:", i, "Cross entropy cost:", cost/N*n_dimension, "Correct rate:", (float(n_correct)/n_total_words * 100.))

                prediction = output_to_text(p)
                print("last output ", prediction)

                costs.append(cost)

                file = open("progress.txt", "a") 
                sp = ' '.join(prediction)
                l = ["\n\nEpoch: ", str(i), sp]
                s = ' '.join(l)
                file.write(s)
                file.close()
                
                
        # save network weights
        self.save_weights()

        #plt.plot(costs)
        #plt.show()
        




 
#################################################################################
# train network
################################################################################
# init network
rnn = RNN(n_dimension, n_hidden, n_vocabulary)

# nuke the last data file
if os.path.isfile('progress.txt'):
    os.remove('progress.txt')


# train network
rnn.fit(input)

#generate_text(rnn)

