import gzip
import pickle
import urllib
import os
import random

from os.path import isfile




def atisfull():
    f = 'atis.pkl'
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
    f = 'atis.pkl'
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
 
"""
def load():
        
    w2ne, w2la = {}, {}
    
    train, test, dic = atisfull()    
    #train, test, dic = atisfold(1)
    
    w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']


    idx2w  = dict((v,k) for k,v in w2idx.items())
    idx2ne = dict((v,k) for k,v in ne2idx.items())
    idx2la = dict((v,k) for k,v in labels2idx.items())

    t_x,  t_ne,  t_label  = test
    train_x, train_ne, train_label = train
    wlength = 35

    test_x = t_x[0:400]
    valid_x = t_x[401:-1]
    
    test_ne = t_ne[0:400]
    valid_ne = t_ne[401:-1]
    
    test_label = t_label[0:400]
    valid_label = t_label[401:-1]
    
    print(len(test_x), len(valid_x))
    print(test_x[0])
    print(valid_x[0])

    
   

    
    for e in ['train','test']:
      for sw, se, sl in zip(eval(e+'_x'), eval(e+'_ne'), eval(e+'_label')):
        print ('WORD'.rjust(wlength), 'LABEL'.rjust(wlength))
        for wx, la in zip(sw, sl): print (idx2w[wx].rjust(wlength), idx2la[la].rjust(wlength))
        print ('\n'+'**'*30+'\n')
"""


