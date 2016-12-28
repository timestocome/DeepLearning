import random


# shuffle data in place
def shuffle(lol, seed):

    for l in lol:
        random.seed(seed)
        random.shuffle(l)



# break data into minibatch
def minibatch(l, bs):
    
    out  = [l[:i] for i in range(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in range(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out



# create a matrix from our input vector
# shift the word to the right
# pad left with -1
def contextwin(l, window_size):
    
    assert (window_size % 2) == 1           # window size must be odd
    assert window_size >=1                  # window size must be 3 or greater
    l = list(l)                             # array of word indexes

    lpadded = window_size//2 * [-1] + l + window_size//2 * [-1] # pad beginning and end with -1
    out = [ lpadded[i:i+window_size] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

