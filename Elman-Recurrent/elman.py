import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

# Elman recurrent network


class model(object):
    
    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # add one for padding 
        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ne+1, de)).astype(theano.config.floatX))        # input layer
                   
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))     # input to hidden weights
                   
                   
        self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))          # hidden to hidden weights
                   
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))          # hidden to output weights
                   
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))   # hidden weights bias
        
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX)) # ? output bias ?
        
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))  # hidden layer

        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        
        # build input from embedding matrix
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y    = T.iscalar('y') # label - target

        # input * input weights + previous hidden * hidden weights + bias
        # then softmax so we have output probabilities
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]
        
        # loop over input matrix processing it through recurrence equations
        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])

        # tag end of sentence
        p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        
        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.log(p_y_given_x_lastword)[y]
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        
        # compile theano functions
        # make a prediction
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        # train network
        self.train = theano.function( inputs  = [idxs, y, lr],
                                      outputs = nll,
                                      updates = updates )
                                      
                                      
        # keep word embeddings on unit sphere by normalizing after each update
        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})


    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
