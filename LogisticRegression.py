#/python

# built in
import numpy as np
import pickle, gzip
import timeit
import sys


# 3rd party
import theano 
import theano.tensor as T


# myy stuff
import LoadData




#################################################################################################
# constants 
##################################################################################################
learning_rate = 0.13
n_epochs = 1000
batch_size = 600

#################################################################################################
# load up data 
##################################################################################################
 
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


datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
   
# set up datasets
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]
    
# set up constants
# // divide and round down to floor
n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
        
         
##########################################################################################
# Logistic Regression 
##########################################################################################
class LogisticRegression(object):

    def __init__(self, input, n_inputs, n_outputs):
    
        # initialize parameters 
        self.W = theano.shared(value=np.zeros((n_inputs, n_outputs), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_outputs,), dtype=theano.config.floatX), name='b', borrow=True)
   
        # map input to hyperplane to determine output
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
   
        # compute predicted class
        self.y_predict = T.argmax(self.p_y_given_x, axis=1)
   
        self.params = [self.W, self.b]
        self.input = input
   


    # using mean instead of sum allows for different batch sizes with out adjusting learning rate
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    

    # number of errors in the mini batch
    def errors(self, y):
        return T.mean(T.neq(self.y_predict, y))
    
    
    def sgd_optimization_mnist():    
    
     
    
        # build the model
        print("building model...")
    
        # setup variables
        index = T.lscalar()     # index to a mini batch
        x = T.matrix('x')       # image data
        y = T.ivector('y')      # labels
    

        classifier = LogisticRegression(input=x, n_inputs=28*28, n_outputs=10)
    
        cost = classifier.negative_log_likelihood(y)
    
        test_model = theano.function(inputs=[index], outputs=classifier.errors(y), 
                                givens={    x: test_set_x[index * batch_size:(index+1) * batch_size],
                                            y: test_set_y[index * batch_size:(index+1) * batch_size]
                                            })
                                            
        validate_model = theano.function(inputs=[index], outputs=classifier.errors(y),
                                givens={     x: valid_set_x[index * batch_size:(index+1) * batch_size],
                                             y: valid_set_y[index * batch_size:(index+1) * batch_size]
                                             })
                                                 
        # compute gradient of cost with respect to theta = (W, b)
        g_W = T.grad(cost=cost, wrt=classifier.W)
        g_b = T.grad(cost=cost, wrt=classifier.b)
    
        # update weights and biases
        updates = [(classifier.W, classifier.W - learning_rate * g_W),
                (classifier.b, classifier.b - learning_rate * g_b)]
                
        # training model description
        train_model = theano.function( inputs=[index], outputs=cost, updates=updates, 
                                givens={       x: train_set_x[index * batch_size:(index+1) * batch_size],
                                               y: train_set_y[index * batch_size:(index+1) * batch_size]
                                               })
                                               
        print("Training model.........")
        # early stopping test values
        patience = 5000                 # minimum number of examples to use
        patience_increase = 2           # wait at least this long before updtaeing best
        improvement_threshold = 0.995   # min significant improvement
        validation_frequency = min(n_train_batches, patience)
        best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()
        done_looping = False
        epoch = 0
    
        while ( epoch < n_epochs ) and ( not done_looping ):        
            epoch = epoch + 1
        
            for minibatch_index in range(n_train_batches):
                
                minibatch_avg_cost = train_model(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index
                
                if (iter + 1) % validation_frequency == 0:
                
                    validation_losses = [validate_model(i) for i in range(n_valid_batches)]

                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % 
                            (epoch, minibatch_index+1, n_train_batches, this_validation_loss*100.))
                            
                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        
                        best_validation_loss = this_validation_loss
                    
                        test_losses = [test_model(i) for i in range(n_test_batches)]
                        test_score = np.mean(test_losses)
                    
                        print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') % 
                                    (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
                                
                        with open('best_model.pkl', 'wb') as f:
                            pickle.dump(classifier, f)
            
                if patience <= iter:
                    done_looping = True
                    break
                
        end_time = timeit.default_timer()    
        print(('Optimization complete with best validation score of %f %%, with test performance %f %%') %
                    (best_validation_loss * 100., test_score * 100.))
                    
        print(('The code ran for %d epochs, with %f epochs/sec' % 
                    (epoch, 1. * epoch / (end_time - start_time))), file=sys.stderr)
                    
  
  
    def predict():                  

        classifier = pickle.load(open('best_model.pkl'))
    
        predict_model = theano.function(inputs=[classifier.input], outputs=classifier.y_pred)
    
        test_x, test_y = datasets[1]
        validate_x, validate_y = datasets[2]
    
        predicted_test_values = predict_model(test_x)
        predicted_validate_values = predict_model(validate_x)
    
        ########
        #compare predicted to correct labels and print accuracy for both sets
    
    
    
    
    
    
###########################################################################################
# create, train, save network
LogisticRegression.sgd_optimization_mnist()

# make predictions, check accuracy


