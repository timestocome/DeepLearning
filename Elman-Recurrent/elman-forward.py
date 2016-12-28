import numpy
import time
import sys
import subprocess
import os
import random

import load
from elman import model
from tools import shuffle, minibatch, contextwin



# recurrent net with memory
# started with code from Deep Learning,
# converted it from Python 2.x to 3.x
# removed PERL script and hacked in inefficient but working accuracy calculation in its place


if __name__ == '__main__':

    s = {'fold':3, # 5 folds 0,1,2,3,4
         'lr':0.0627142536696559,       # learning rate - try 0.05->0.01
         'verbose':1,
         'decay':False,                 # decay on the learning rate if improvement stops
         'win':7,                       # number of words in the context window try 3->19
         'bs':9,                        # number of backprop through time steps
         'nhidden':100,                 # number of hidden units try 100->200
         'seed':345,
         'emb_dimension':100,           # dimension of word embedding try 50->100
         'nepochs':1}

    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    train_set, valid_set, test_set, dic = load.atisfold(s['fold'])
    idx2label = dict((k,v) for v,k in dic['labels2idx'].items())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].items())

    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex,  test_ne,  test_y  = test_set

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    # describe model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    ne = vocsize,
                    de = s['emb_dimension'],
                    cs = s['win'] )


    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s['clr'] = s['lr']
    for e in range(s['nepochs']):
    
        # shuffle training data
        shuffle([train_lex, train_ne, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()
        
        # for each sentence
        for i in range(nsentences):
            cwords = contextwin(train_lex[i], s['win'])
            words  = map(lambda x: numpy.asarray(x).astype('int32'), minibatch(cwords, s['bs']))
            labels = train_y[i]
            
            # each sentence is a minibatch, perform one update per sentence
            for word_batch , label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, s['clr'])
                rnn.normalize()
                
            print ('training epoch %i > %2.2f%%' % (e,(i+1)*100./nsentences))
            sys.stdout.flush()

       

        
        # evaluation
        # test_y, valid_y, train_y contain correct labels
        count = 0
        running_average = 0.0
        for x in valid_lex:
            test_output = rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32'))
            a = test_output
            b = valid_y[count][:]
            if len(a) < len(b):
                b = b[0:len(a)]
            else:
                a = a[0:len(b)]
       
            correct = sum(a == b)
            running_average += correct/len(a)
            count += 1
        print ("Average correct", running_average/len(valid_lex))
        
        # learning rate decay if no improvement in 10 epochs
        if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5
        if s['clr'] < 1e-5: break


        # info to user
        #test_output_words = [idx2label for z in a]
        #print("test_output", test_output_words)

        