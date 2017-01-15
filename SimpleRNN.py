# http://github.com/timestocome
# converted to python 3, streamlined code, clarified code
# added init weights into init function
# fixed and streamlined backpropagation update function
# adjusted parameters for faster, better accuracy
# removed unused variables and code
# added comments for clarity


# adapted from 
# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python



import numpy as np

import theano
import theano.tensor as T 

import matplotlib.pyplot as plt

from sklearn.utils import shuffle


# setup theano on GPU if possible
GPU = True
if GPU:
    print("Device set to GPU")
    try: theano.config.device = 'gpu'
    except: pass    # its already set
    theano.config.floatX = 'float32'
else:
    print("Running with CPU")



rng = np.random.RandomState(27)     # prime random number generator


#  Network parameters
# collect them all up here so it's easier to adjust them
n_hidden = 4
n_out = 2
learning_rate = 10e-4
epochs = 20
nbit = 12



# create data to use in training
# creates all possible binary combinations for nbits
def create_parity_pairs():

    n = 2 **nbit                # number of possible combinations
    x = np.zeros((n, nbit))     
    y = np.zeros(n)

    for i in range(n):
        for j in range(nbit):

            if i % (2 **(j+1)) != 0:
                i -= 2 **j
                x[i,j] = 1
                y[i] = x[i].sum() % 2

    x = x.reshape(n, x.shape[1], 1)
    y = y.reshape(y.shape[0], 1)

    return x.astype('float32'), y.astype('int32')



class SimpleRNN:

    def __init__(self, number_samples):
        

        # set up weights and biases
        d = 1                           # depth of x               
        n = number_samples
    
        # init
        Wx = np.asarray(rng.uniform(
            low = -np.sqrt(2. /(d + n_hidden)),
            high = np.sqrt(2. /(d + n_hidden)),
            size = (d, n_hidden)
        ))
        self.Wx = theano.shared(Wx, name='Wx', borrow=True)

        Wh = np.asarray(rng.uniform(
            low = -np.sqrt(2. /(d + n_hidden)),
            high = np.sqrt(2 /(d + n_hidden)),
            size = (n_hidden, n_hidden)
        ))       
        self.Wh = theano.shared(Wh, name='Wh', borrow=True)

        bh = np.zeros(n_hidden)
        self.bh = theano.shared(bh, name='bh', borrow=True)

        ho = np.zeros(n_hidden)
        self.ho = theano.shared(ho, name='ho', borrow=True)

        Wo = np.asarray(rng.uniform(
            low = -np.sqrt(2. /(n_hidden + n_out)),
            high = np.sqrt(2. /(n_hidden + n_out)),
            size = (n_hidden, n_out)
        )) 
        self.Wo = theano.shared(Wo, name='Wo', borrow=True)

        bo = np.zeros(n_out)
        self.bo = theano.shared(bo, name='bo', borrow=True)

        # values to adjust with back propagation
        self.parameters = [self.Wx, self.Wh, self.bh, self.ho, self.Wo, self.bo]

        # recurrence functions
        thX = T.fmatrix('x')
        thY = T.ivector('y')

        # feed forward equations
        def recurrence(x_t, h_t1):
            h_t = T.nnet.relu( T.dot(x_t, self.Wx) + T.dot(h_t1, self.Wh) + self.bh )
            y_t = T.nnet.softmax( T.dot(h_t, self.Wo) + self.bo )
            return h_t, y_t

        # loop over feed forward equations once for each bit in the sequence 
        # send previous hidden output back through and collect prediction
        [h, y_predicted], _ = theano.scan(
            fn = recurrence,
            outputs_info = [self.ho, None],
            sequences = thX,
            n_steps = thX.shape[0],
        )

        # probability of x given y
        py_x = y_predicted[:, 0, :]
        prediction = T.argmax(py_x, axis=1) # fetch most likely prediction

        # cost functions for gradients and tracking progress
        cost = -T.mean( T.log(py_x[T.arange(thY.shape[0]), thY]))       # cross entropy
        gradients = T.grad(cost, self.parameters)                       # derivatives
       
        updates = [(p, p - learning_rate * g) for p, g in zip(self.parameters, gradients)]


        # training and prediction functions
        self.predict_op = theano.function(inputs = [thX], outputs = prediction)

        self.train_op = theano.function(
                    inputs = [thX, thY],
                    outputs = cost,
                    updates = updates
        )


    def train(self, x, y):

        costs = []

        # number of times to loop through all of data set
        for i in range(epochs):
            
            x, y = shuffle(x, y)            # things work better when you shuffle the data
            cost = 0

            for j in range(len(y)):

                c = self.train_op(x[j], y[j])
                cost += c 
            
            # output cost so user can see training progress
            cost /= len(y)
            print ("i:", i, "cost:", cost, "%")
            costs.append(cost)
            
        # graph to show accuracy progress - cost function should decrease
        plt.plot(costs)
        plt.show()




def parity():
    # create training data
    x,y = create_parity_pairs()
    
    # create and train network
    rnn = SimpleRNN(len(y))
    rnn.train(x, y)
    

parity()