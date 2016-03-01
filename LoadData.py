#!/python


import numpy as np
import pandas as pd
import pickle

import LoadRawMNISTData

    
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
   
   testing_images, testing_labels = LoadRawMNISTData.load_mnist("testing")
   print("testing ? ", len(testing_labels))
   training_images, training_labels = LoadRawMNISTData.load_mnist("training")
   print("training ? ", len(training_labels))


   # convert image data from 0-255 ints to 0.0-1.0 floats
   testing_images = testing_images / 255.0
   training_images = training_images / 255.0


   # zip image data and label data together
   testing_data = zip(testing_images, testing_labels)
   training_data = zip(training_images, training_labels)

   
   # write data out to files
   pickle_out = open('test.pkl', 'wb')
   pickle.dump(testing_data, pickle_out)
   pickle_out.close() 
   
   pickle_out = open('train.pkl', 'wb')
   pickle.dump(training_data, pickle_out)
   pickle_out.close() 
   
   
   

# uncomment line below to convert the raw files into pickle files
# once you've written the pickle files there's no need to rerun this function    
#csv_to_pickle()




# load up data from pkl file
def load_data():
   
    file = 'train.pkl'
    with open(file, 'rb') as f:
        training_data = pickle.load(f)
        f.close()
   
    training = list(training_data)

        
    file = 'test.pkl'
    with open(file, 'rb') as f:
        testing_data = pickle.load(f)
        f.close()     
    testing = list(testing_data)   
     
   
        
    return (training, testing)
    

   
    
# re-shape data to match network requirements
def load_data_wrapper():
    tr_d, te_d = load_data()
     
    tr_i, tr_l = zip(*tr_d)
    tr_image = np.asarray(tr_i)
    tr_label = np.asarray(tr_l)
    
    te_i, te_l = zip(*te_d)
    te_image = np.asarray(te_i)
    te_label = np.asarray(te_l)
    
   
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_image]
    training_results = [vectorized_result(y) for y in tr_label]
    training_data = zip(training_inputs, training_results)
      
 
    test_inputs = [np.reshape(x, (784, 1)) for x in te_image]
    test_data = zip(test_inputs, te_label)
        
    return(tr_image, tr_label, te_image, te_label)     # return images and labels
    #return (training_data, test_data)                   # return combined image/labels
   
  
    
    

# convert 0-9 labels to 10 zero arrays with a 1 in the correct position
def vectorized_result(j):
    
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


    

# read in data    
training_data, test_data = load_data()
print("Successfully loaded training/testing?", len(training_data), len(test_data))

