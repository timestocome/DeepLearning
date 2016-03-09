#!/python


import numpy as np
import pandas as pd
import pickle



    
# 28x28 images 0-783 as ints
# training set has 42000 images [label, pixel0, pixel1, ... , pixel783]
# testing set has 28000 images[pixel0, pixel1, ... , pixel783]




# easier to read in csv and store as pickle file, only need to do this once
# submission.pkl is the validation data whose answers we'll submit to kaggle
#
# break the training.csv file into training and testing data
#
# can reshuffle testing and training data by re-running this function or change
# sizes of test/train. 
def csv_to_pickle():
   
   # validation data to submit to Kaggle
   k = pd.read_csv('test.csv')

   
   # training data - need to break this into training, testing, validation
   data = pd.read_csv('train.csv')
   
   
   # shuffle data so we get random items in each set
   shuffled_data = np.random.permutation(data.index)
   testing = (data[0:1000]).as_matrix()
   validation = (data[1000:2000]).as_matrix()
   training = (data[2000:42000]).as_matrix()
   kaggle = k.as_matrix()
   
   
   # grab labels
   testing_labels = testing[:, 0]
   validation_labels = validation[:, 0]
   training_labels = training[:,0]
   kaggle_labels = np.zeros((28000), dtype=np.int)          # placeholder to keep data groups all in same format
    

    
   # grab image data arrays of 784
   testing_images = testing[:, 0:784]
   validation_images = validation[:, 0:784]
   training_images = training[:, 0:784]
   kaggle_images = kaggle[:, 0:784]
    


   # convert image data from 0-255 ints to 0.0-1.0 floats
   testing_images = testing_images / 255.0
   training_images = training_images / 255.0
   validation_images = validation_images / 255.0
   kaggle_images = kaggle_images / 255.0

   # zip image data and label data together
   testing_data = zip(testing_images, testing_labels)
   validation_data = zip(validation_images, validation_labels)
   training_data = zip(training_images, training_labels)
   kaggle_data = zip(kaggle_images, kaggle_labels)
   
   # write data out ot files
   pickle_out = open('validate.pkl', 'wb')
   pickle.dump(validation_data, pickle_out)
   pickle_out.close() 
   
   pickle_out = open('test.pkl', 'wb')
   pickle.dump(testing_data, pickle_out)
   pickle_out.close() 
   
   pickle_out = open('train.pkl', 'wb')
   pickle.dump(training_data, pickle_out)
   pickle_out.close() 
   
   pickle_out = open('submission.pkl', 'wb')
   pickle.dump(kaggle_data, pickle_out)
   pickle_out.close()
   

# uncomment line below to convert the csv files into pickle files    
#csv_to_pickle()




# load up data from pkl file
def load_data():
   
    file = 'mnist_expanded.pkl'
    #file = 'train.pkl'
    with open(file, 'rb') as f:
        training_data = pickle.load(f)
        f.close()
   
    training = list(training_data)

        
    file = 'test.pkl'
    with open(file, 'rb') as f:
        testing_data = pickle.load(f)
        f.close()     
    testing = list(testing_data)   
     
    file = 'validate.pkl'
    with open(file, 'rb') as f:
        validation_data = pickle.load(f)
        f.close()      
    validate = list(validation_data) 
     
     
    # kaggle validation data
    file = 'submission.pkl'
    with open(file, 'rb') as f:
        kaggle = pickle.load(f)
        f.close()
       
        
    return (training, validate, testing, kaggle)
    

   
    
# re-shape data to match network requirements
def load_data_wrapper():
    tr_d, va_d, te_d, ka_d = load_data()
    
    # expanded dataset
    tr_i = tr_d[0]
    tr_l = tr_d[1]
    
    # or 
    # original data set
    #tr_i, tr_l = zip(*tr_d)
    
    tr_image = np.asarray(tr_i)
    tr_label = np.asarray(tr_l)
    
    
    
    va_i, va_l = zip(*va_d)
    va_image = np.asarray(va_i)
    va_label = np.asarray(va_l)
    
    te_i, te_l = zip(*te_d)
    te_image = np.asarray(te_i)
    te_label = np.asarray(te_l)
    
    ka_i, ka_l = zip(*ka_d)
    ka_image = np.asarray(ka_i)
    ka_label = np.asarray(ka_l)
  
   
    
    
    training_inputs = [np.reshape(x, (784)) for x in tr_image]
    training_data = zip(training_inputs, tr_label)
    
    validation_inputs = [np.reshape(x, (784)) for x in va_image]
    validation_data = zip(validation_inputs, va_label)
    
    test_inputs = [np.reshape(x, (784)) for x in te_image]
    test_data = zip(test_inputs, te_label)
    
    kaggle_inputs = [np.reshape(x, (784)) for x in ka_image]
    kaggle_data = zip(kaggle_inputs, ka_label)
    
    return (training_data, validation_data, test_data, kaggle_data)
   
  
    
    


# read in data    
training_data, validation_data, test_data, kaggle_data = load_data()




