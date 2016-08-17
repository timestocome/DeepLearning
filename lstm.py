
# https://github.com/timestocome/MiscDeepLearning
# LSTM to read in a sequence of numbers and predict next digit in series
# achieve about 70% accuracy with only 60 digits in knights move training set
#
# I started with this code
# http://deeplearning.net/tutorial/code/lstm.py
# starter code contains several different learning and utility functions 
# I only kept the ones I'm currently using to simplify the code

# lstm 
# used for predicting next word or char in a sequence
# sentiment language analysis, DSP separation of signals
# sequence to sequence ( sentence in one language to sentence in a different language)
#    in s to s reversing the word order on the input sentence worked better ? idk

# completed 
# change code to read in sequence of digits and predict next digit
# clean up code to be more readable
# added better comments, links to more information
# sped up code
# removed keyboard interrupt, not needed
# fix test/valid indexes not resetting in epoch loop
# remove more of the unused code 

# to do
# why is train function not seeing some of the global vars?
# regularization or clipping needed? 


from collections import OrderedDict
import sys
import time
import pickle

import numpy as np

import theano
from theano import config
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams



######################################################################################################
# network variables
######################################################################################################
max_epochs = 2000              # number of times to loop entire training set
batch_size = 4                # number of training samples used per training loop
lstm_depth = 3

np.random.seed(27)            # use same seed for random while testing
trng = RandomStreams(27)

dim_proj = 64                # word embeding dimension and LSTM number of hidden units.

patience = max_epochs/10      # Number of epoch to wait before early stop if no progress
dispFreq = 10                 # Display to stdout the training progress every N updates

decay_c = 0.                  # Weight decay for the classifier applied to the U weights.
lrate = 0.0001                # Learning rate for sgd (not used for adadelta and rmsprop)

n_input = 1000                # maximum value for input or output number in sequence
n_output = n_input

validFreq = 100               # Compute the validation error after this number of update.

noise_std = 0.,               # used for dropout
use_dropout = True            # if False slightly faster, but worst test error


reload_model = None           # Path to a saved model we want to start from.
saveto = 'lstm_model.npz'     # The best model will be saved there
saveFreq = 100                # Save the parameters after every saveFreq updates

print("finished init globals")




#####################################################################################################
# utilities for saving and reloading best parameters
#####################################################################################################
# used to reload model
def zipp(params, tparams):
   
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


# used to save model
def unzip(zipped):
   
    new_params = OrderedDict()

    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    
    return new_params



#####################################################################################################
# data load, shuffle ....
# really need much larger set for training but this will do for devolopment and testing.
#####################################################################################################
# knights move pascal triangle
knights_move =  [1, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 4, 4, 2, 1, 1, 5, 7, 4, 2, 1, 1, 6, 11, 8, 4, 2, 1, 1, 7, 16, 15, 8, 4, 2, 1, 1, 8, 22, 26, 16, 8, 4, 2, 1, 1, 9, 29, 42, 31, 16, 8, 4, 2, 1, 1, 10, 37, 64, 57, 32, 16, 8, 4, 2, 1, 1, 11, 46, 93, 99, 63, 32, 16, 8, 4, 2, 1]

print("Knights move", knights_move)
print("***************************")
    
sequence_length = len(knights_move)
   
# create training, test, validation sets
x_sets = []
y_sets = []
x_test_sets = []
y_test_sets = []
x_valid_sets = []
y_valid_sets = []
count = 0    
    
total_sets = sequence_length - lstm_depth - 1
n_valid = total_sets // 10
n_test = n_valid
n_train = total_sets - n_valid * 2


for i in range(sequence_length - lstm_depth - 1):
        input = knights_move[i:i+lstm_depth]
        output = knights_move[i+lstm_depth + 1]
        count += 1

        if count <= n_train:
            x_sets.append(input)
            y_sets.append(output)
        elif count <= n_train + n_test:
            x_test_sets.append(input)
            y_test_sets.append(output)
        else:
            x_valid_sets.append(input)
            y_valid_sets.append(output)
   
print(len(y_sets), total_sets)
print(len(y_sets), len(y_test_sets), len(x_valid_sets))


# find max number and set output vector to that size plus a little
n_output = np.max(y_sets) + 1

# adjust our inputs up if the max integer in series is larger, smaller than expected
if n_output < n_input:
    n_output = n_input
else:
    n_input = n_output


# swap axis on data set 
def prepare_data(seqs, labels, n_input=None):
  
    # x: a list of integers
    lengths = [len(s) for s in seqs]

    if n_input is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []

        for l, s, y in zip(lengths, seqs, labels):
            if l < n_input:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    n_input = np.max(lengths)

    x = np.zeros((n_input, n_samples)).astype('int64')
    x_mask = np.zeros((n_input, n_samples)).astype(theano.config.floatX)

    # x_mask puts ones where data exists in x, 0s where there is only padding 
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.
        
    return x, x_mask, labels



# shuffle data
def get_minibatches_idx(n, minibatch_size, shuffle=False):
   
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0

    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start : minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    # Make a minibatch out of what is left
    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)





######################################################################################################
# setup weights, bias, convert to shared variables for theano
#######################################################################################################

# create and initialize word embedding
# take each input vector, extract information from each item and embed into semantic vector
#    used to capture meaning of a sentence. Not sure this is useful with numeric sequences?
#    original code did sentiment anaylsis  
# https://arxiv.org/pdf/1502.06922.pdf
def init_params():

    params = OrderedDict()
    
    # embedding
    randn = np.random.rand(n_input, dim_proj)
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    fns = layers['lstm']
    params = fns[0](params)

    # classifier
    params['U'] = 0.01 * np.random.randn(dim_proj, n_output).astype(config.floatX)
    params['b'] = np.zeros((n_output,)).astype(config.floatX)

    return params




# convert created or loaded parameters to theano shared variables
def init_tparams(params):

    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)

    return tparams


# init lstm weights with ortho normalized matrices
# http://arxiv.org/pdf/1312.6120v3.pdf section 1.1
def ortho_weight(ndim):

    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)

    return u.astype(config.floatX)



def param_init_lstm(params): 

    # input gate, forget gate, output gate, recurrent
    W = np.concatenate([ortho_weight(dim_proj),
                           ortho_weight(dim_proj),
                           ortho_weight(dim_proj),
                           ortho_weight(dim_proj)], axis=1)
    params['lstm_W'] = W


    # input gate, forget gate, output gate, recurrent
    U = np.concatenate([ortho_weight(dim_proj),
                           ortho_weight(dim_proj),
                           ortho_weight(dim_proj),
                           ortho_weight(dim_proj)], axis=1)
    params['lstm_U'] = U
    
    # bias for each of 4 parts of memory cell
    b = np.zeros((4 * dim_proj,))
    params['lstm_b'] = b.astype(config.floatX)
    

    return params


######################################################################################################
# LSTM
# http://deeplearning.net/tutorial/lstm.html ( notes for starter code )
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
######################################################################################################

def lstm_layer(tparams, state_below, mask=None):

    nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, tparams['lstm_U'])
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, dim_proj))     # input gate
        f = T.nnet.sigmoid(_slice(preact, 1, dim_proj))     # forget gate
        o = T.nnet.sigmoid(_slice(preact, 2, dim_proj))     # output gate
        c = T.tanh(_slice(preact, 3, dim_proj))             # memory cell

        c = f * c_ + i * c                                  # new state for memory cell
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)                                   # output of cell / node
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (T.dot(state_below, tparams['lstm_W']) + tparams['lstm_b'])

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[T.alloc((np.asarray(0., dtype=config.floatX)), n_samples, dim_proj),
                                              T.alloc((np.asarray(0., dtype=config.floatX)), n_samples, dim_proj)],
                                name='lstm_layers',
                                n_steps=nsteps)
    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}



######################################################################################################
# learning functions
######################################################################################################

# adaptive learning rate optimizer
# https://arxiv.org/abs/1212.5701 AdaDelta, Zeiler
def adadelta(lr, tparams, grads, x, mask, y, cost):
    
    zipped_grads = [theano.shared(p.get_value() * (np.asarray(0., dtype=config.floatX)), name='%s_grad' % k)
                    for k, p in tparams.items()]

    running_up2 = [theano.shared(p.get_value() * (np.asarray(0., dtype=config.floatX)), name='%s_rup2' % k)
                   for k, p in tparams.items()]

    running_grads2 = [theano.shared(p.get_value() * (np.asarray(0., dtype=config.floatX)), name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]

    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]

    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

    return f_grad_shared, f_update



# drop out --- don't count a random selection of half of weights during training updating
# p = % items randomly set to zero
# http://stackoverflow.com/questions/31971462/bernoulli-in-theano
def dropout_layer(state_before, use_noise, trng):

    proj = T.switch(use_noise,
                         (state_before * trng.binomial(state_before.shape,
                          p = 0.5, 
                          n = 1,
                          dtype = state_before.dtype)),
                          state_before * 0.5)
    return proj




######################################################################################################
# define network
######################################################################################################

def build_model(tparams):

    # place holders for data
    x = T.matrix('x', dtype='int64')
    mask = T.matrix('mask', dtype=config.floatX)
    y = T.vector('y', dtype='int64')

    # depth/timesteps = columns in training vectors
    # number of training vectors is rows
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # training vectors are embedded into representations of the input numbers
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, dim_proj])

    # functions
    fns = layers['lstm']

    # 
    proj = fns[1](tparams, emb, mask=mask)
    proj = (proj * mask[:, :, None]).sum(axis=0)
    proj = proj / mask.sum(axis=0)[:, None]
    
    # improved robustness of network, better generalization
    if use_dropout:
        use_noise = theano.shared(np.asarray(0., dtype=config.floatX))
        proj = dropout_layer(proj, use_noise, trng)

    # output from lstm is smoothed and winning item calculated
    pred = T.nnet.softmax(T.dot(proj, tparams['U']) + tparams['b'])
    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    # prevent divide by zero
    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    # error cost used to adjust the weights
    cost = -T.log(pred[T.arange(n_samples), y] + off).mean()


    return use_noise, x, mask, y, f_pred_prob, f_pred, cost






# get predictions on a batch and check against correct value
def pred_error(f_pred, x_in, y_in, iterator, verbose=False):
   
    valid_error = 0
    count = 0
    for i, valid_index in iterator:
        
        x, mask, y = prepare_data([x_in[t] for t in valid_index], np.array(y_in)[valid_index], n_input=None)

        preds = f_pred(x, mask)
        targets = np.array(y)
        valid_error += (preds == targets).sum()
        count += len(preds)
        
        
    if count != 0:   
        valid_error= 1. - valid_error / count
    else:
        valid_error = 0.
    

    return valid_error




def train_lstm():

    decay_c = 0.
    validFreq = 10
    saveFreq = 1000

    # if we have test samples
    if n_test > 0:    
        idx = np.arange(n_test)
        np.random.shuffle(idx)
        idx = idx[:n_test]
        test = ([x_test_sets[n] for n in idx], [y_test_sets[n] for n in idx])


    print('Building model')

    # load up weights, bias, 
    # create or load initial weights, bias ....
    params = init_params()

    if reload_model:
       load_params('lstm_model.npz', params)

    # convert newly created or loaded saved params to shared variables
    tparams = init_tparams(params)


    # set up the model
    # use_noise is for dropout
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(tparams)

   
    # regularization of weights?
    if decay_c > 0.:
        decay_c = theano.shared(value=decay_c, name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # compute cost
    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    # compute gradients
    grads = T.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    # adjust learning rate
    lr = T.scalar(name='lr')
    f_grad_shared, f_update = adadelta(lr, tparams, grads, x, mask, y, cost)


    print('Optimization')

    # track progress    
    history_errs = []
    best_p = None
    bad_count = 0

    update_index = 0                                 # the number of update done
    early_stop = False                               # early stop
    start_time = time.time()


    
    for epoch in range(max_epochs):

        n_samples = 0

        # Get new shuffled index for the training set.
        kf = get_minibatches_idx(n_train, batch_size, shuffle=True)

        for _, train_index in kf:
     
            update_index += 1
            use_noise.set_value(1.)

            # Select the random examples for this minibatch
            y = [y_sets[t] for t in train_index]
            x = [x_sets[t]for t in train_index]

            # Get the data in numpy.ndarray format
            # Then swap the axis!
            # Return something of shape (minibatch n_input, n samples)
            x, mask, y = prepare_data(x, y)
            n_samples += x.shape[1]

            cost = f_grad_shared(x, mask, y)
            f_update(lrate)

            # sanity check
            if np.isnan(cost) or np.isinf(cost):
                print('bad cost detected: ', cost)
                return 1., 1., 1.

            # show user our progress
            if np.mod(update_index, dispFreq) == 0:
               print('Epoch ', epoch, 'Update ', update_index, 'Cost ', cost)

            # occasionally save information
            if saveto and np.mod(update_index, saveFreq) == 0:
                print('Saving...')

                if best_p is not None:
                   params = best_p
                else:
                    params = unzip(tparams)
                np.savez(saveto, history_errs=history_errs, **params)
                print('... finished saving')


            # run test and validation sets through to see progress    
            if np.mod(update_index, validFreq) == 0:
                
                use_noise.set_value(0.)

                train_err = pred_error(f_pred, x_sets, y_sets, kf)

                kf_valid = get_minibatches_idx(n_valid, batch_size)
                kf_test = get_minibatches_idx(n_test, batch_size)

                valid_err = pred_error(f_pred, x_valid_sets, y_valid_sets, kf_valid)
                test_err = pred_error(f_pred, x_test_sets, y_test_sets, kf_test)
                history_errs.append([valid_err, test_err])

                print('Errors: Train %.2lf%%, Valid %.2lf%%, Test %.2lf%%' % (train_err * 100., valid_err * 100., test_err * 100.))
                
                # is this our best run?
                if ( best_p is None or
                    valid_err <= np.array(history_errs)[:,0].min() ):

                    best_p = unzip(tparams)
                    bad_counter = 0

                # bail ?
                if (len(history_errs) > patience and
                    valid_err >= np.array(history_errs)[:-patience, 0].min()):

                    bad_counter += 1
                    if bad_counter > patience:
                        print('Early Stop!')
                        early_stop = True
                        break
            
        print('**************************************************************************')
        if early_stop:
            break

   

    
    # clean up
    end_time = time.time()
    
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    # reset batches and test
    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(n_train, batch_size)
    kf_valid = get_minibatches_idx(n_valid, batch_size)
    kf_test = get_minibatches_idx(n_test, batch_size)

    train_error = pred_error(f_pred, x_sets, y_sets, kf_train_sorted)
    valid_error = pred_error(f_pred, x_valid_sets, y_valid_sets, kf_valid)
    test_error = pred_error(f_pred, x_test_sets, y_test_sets, kf_test)

    print( 'Final errors: Train %.2lf%%, Valid %.2lf%%, Test %.2lf%%' % ( train_error * 100., valid_error * 100., test_error * 100.))


    if saveto:
        np.savez(saveto, train_err=train_error, valid_err=valid_error, test_err=test_error, history_errs=history_errs, **best_p)


    print( 'Training took %.1fs' % (end_time - start_time))

    return train_error, valid_error, test_error



######################################################################################################
# Run code
######################################################################################################
train_lstm()




"""
#####################################################################################################
# utility functions for saving and reloading model
#####################################################################################################

# reload parameters from saved model
def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params



# If you want to use a trained model, this is useful to compute
# the probabilities of new examples.
def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
        
    n_samples = len(data[0])
    probs = np.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index],
                                  n_input=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs
"""