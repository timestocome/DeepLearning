import random



def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)




def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in range(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in range(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out




def contextwin(l, window_size):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (window_size % 2) == 1           # window size must be odd
    assert window_size >=1                  # window size must be 3 or greater
    l = list(l)                             # array of word indexes

    lpadded = window_size//2 * [-1] + l + window_size//2 * [-1] # pad beginning and end with -1
    out = [ lpadded[i:i+window_size] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

