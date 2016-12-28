import gzip
import pickle
import urllib
import os
import random

from os.path import isfile

file = 'Elman-Recurrent/atis.pkl'


# read in and load up the data sets

def atisfull():
    f = file
    with open(f, 'rb') as f:
        train_set, test_set, dicts = pickle.load(f, encoding='latin1')
        f.close()
        
    # break test in test and valid 
    t_x,  t_ne,  t_label  = test_set
    test_x = t_x[0:400]
    valid_x = t_x[401:-1]
    
    test_ne = t_ne[0:400]
    valid_ne = t_ne[401:-1]
    
    test_label = t_label[0:400]
    valid_label = t_label[401:-1]
    
    test_set = (test_x, test_ne, test_label)
    valid_set = (valid_x, valid_ne, valid_label)
    
    return train_set, valid_set, test_set, dicts


def atisfold(fold):
    assert fold in range(5)
    f = file
    with open(f, 'rb') as f:
        train_set, test_set, dicts = pickle.load(f, encoding='latin1')
        f.close()
        
        
        # break test in test and valid 
    t_x,  t_ne,  t_label  = test_set
    test_x = t_x[0:400]
    valid_x = t_x[401:-1]
    
    test_ne = t_ne[0:400]
    valid_ne = t_ne[401:-1]
    
    test_label = t_label[0:400]
    valid_label = t_label[401:-1]
    
    test_set = (test_x, test_ne, test_label)
    valid_set = (valid_x, valid_ne, valid_label)
    
    return train_set, valid_set, test_set, dicts
 
