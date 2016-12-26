
# http://github.com/timestocome

# adapted from https://github.com/lisa-lab/DeepLearningTutorials

import pickle
import gzip

import numpy as np
import os
import sys
import timeit

import theano
import theano.tensor as T




# setup theano
GPU = True
if GPU:
    print("Device set to GPU")
    try: theano.config.device = 'gpu'
    except: pass    # its already set
    theano.config.floatX = 'float32'
else:
    print("Running with CPU")



# tuning values -- no reason to keep passing constants into a function
learning_rate = 0.13
n_epochs = 5000
batch_size = 20
  

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

  

###################################################################################
# Multi-class Logistic Regression
####################################################################################

class LogisiticRegression(object):

    def __init__(self, input, n_in, n_out):

        # init weights and bias to zero 
        # shared loads them onto gpu
        # borrow means they get updated immediately
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out), dtype=theano.config.floatX), name='b', borrow=True)

        # the equations for this layer
        # compute probability of y given x
        # predict y given x - axis 1 is the column representing our output
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.input = input

    # y[0] is the number of examples (rows) in our mini-batch
    # columns are our output classes
    # using mean, could use sum/mini-batch-count
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    # count the number of classes we missed on this mini-batch and return the mean
    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))


########################################################################################
# Stochastic gradient descent optimization
########################################################################################

def sgd():

    # build the model
    index = T.lscalar()         # mini-batch index

    x = T.matrix('x')           # input data
    y = T.ivector('y')          # target labels

    input_size = 28 * 28        # image height, width, depth=1
    number_labels = 10          # 0-9

    # set up theano functions
    classifier = LogisiticRegression(x, n_in=input_size, n_out=number_labels)
    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(inputs=[index], outputs=classifier.errors(y), 
                givens={ x: test_x[index * batch_size:(index+1) * batch_size],
                         y: test_y[index * batch_size:(index+1) * batch_size]
                })

    validate_model = theano.function(inputs=[index], outputs=classifier.errors(y),
                givens={ x: valid_x[index * batch_size:(index+1) * batch_size],
                         y: valid_y[index * batch_size:(index+1) * batch_size] 
                })

    dW = T.grad(cost=cost, wrt=classifier.W)
    db = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * dW),
               (classifier.b, classifier.b - learning_rate * db)]

    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
            givens={ x: train_x[index * batch_size:(index+1) * batch_size],
                     y: train_y[index * batch_size:(index+1) * batch_size]
            })


    # train the model
    validation_frequency = 1000     # how often to test validation examples
    best_validation_loss = np.inf   # best score on validation examples
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 1
    target_score = 0.0825
    target_hit = False

    while epoch < n_epochs and target_hit==False:

        epoch += 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)

            if epoch % validation_frequency == 0:

                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i, validation error %f %%' % (epoch, minibatch_index, this_validation_loss * 100.))
                    
                # if best run so far?
                if this_validation_loss < best_validation_loss:
                    best_validation_loss = this_validation_loss

                    # try test ( hold out data )
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print("Best error on hold out data %f %%" % (test_score * 100.) )

                    if test_score < target_score:
                        # save model
                        with open('best_model.pkl', 'wb') as f:
                            pickle.dump(classifier, f)
                            target_hit  = True
                        break
                

    end_time = timeit.default_timer()


    print("Optimization complete ")
    print("Best validation loss ", best_validation_loss * 100.)
    print("Best hold out loss ", test_score * 100.)
    print("Run time ", (end_time - start_time))


def predict():

    # load the saved model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test 
    #dataset='mnist.pkl.gz'
    #datasets = load_data(dataset)
    #test_set_x, test_set_y = datasets[2]
    #test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)



if __name__ == '__main__':
    sgd()