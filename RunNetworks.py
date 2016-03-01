#/python

# built in
import numpy as np
import pickle

# 3rd party
import theano as T


# myy stuff
import LoadData


# read in, format, and load the data files
training_x, training_y, testing_x, testing_y = LoadData.load_data_wrapper()

# break testing into testing and validation

validating_x = testing_x[0:5000]
validating_y = testing_y[0:5000]
testing_x = testing_x[5000:10000]
testing_y = testing_y[5000:10000]




# share the data as floats to store it all on the gpu
train_x = T.shared(np.asarray(training_x, dtype=T.config.floatX))
train_y = T.shared(np.asarray(training_y, dtype=T.config.floatX)) 

test_x = T.shared(np.asarray(testing_x, dtype=T.config.floatX))
test_y = T.shared(np.asarray(testing_y, dtype=T.config.floatX))

validate_x = T.shared(np.asarray(validating_x, dtype=T.config.floatX))
validate_y = T.shared(np.asarray(validating_y, dtype=T.config.floatX))



print("Training, testing, validating data", len(training_y), len(testing_y), len(validating_y))
