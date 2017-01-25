
# http://github.com/timestocome


import numpy as np
import pickle
from collections import Counter


################################################################################
# import and parse data
################################################################################


# open file and read in text
#file = open('AliceInWonderland.txt', 'r')
file = open('BothBooks.txt', encoding='utf-8')
data = file.read()
file.close()



# convert all text to lower case and remove new line chars
data = data.lower()
data = data.replace('\n', ' ')
# the books use ---, ', : so often many words appear only once - stripping them (for now)
# unique words 6591
data = data.replace('--', ' ')  # uniques 6111
data = data.replace(':', ' ')   # uniques 5858
data = data.replace("'", ' ')   # uniques 5389
data = data.replace(',', ' ')   # uniques 4269
data = data.replace(')', ' ')
data = data.replace('(', ' ')   # uniques 4180
data = data.replace('"', ' ')   # uniques 4134
data = data.replace('-', ' ')   # uniques 4057
data = data.replace('[', ' ')
data = data.replace(']', ' ')
data = data.replace('0', ' ')
data = data.replace('1', ' ')
data = data.replace('2', ' ')
data = data.replace('3', ' ')
data = data.replace('4', ' ')
data = data.replace('5', ' ')
data = data.replace('6', ' ')
data = data.replace('7', ' ')
data = data.replace('8', ' ')
data = data.replace('9', ' ')   # uniques 4041


# replace sentence stop/start with flags
data = data.replace('.', ' END_SENTENCE ')
data = data.replace('?', ' END_SENTENCE ')
data = data.replace('!', ' END_SENTENCE ')


#print(data)

# parse the data 
sentences = []
word2idx = {'START':0, 'END':1}
idx2word = ['START', 'END']
current_idx = 2
all_words = []

words = data.split()
new_sentence = ['START']

for w in words:

    # split into sentences
    if w != 'END_SENTENCE':
        new_sentence.append(w)

        # add to word indexes if it's not already there
        if w not in idx2word:
            idx2word.append(w)
            word2idx.update({w : current_idx})
            current_idx += 1
        all_words.append(w)
    else:
        new_sentence.append('END')
        sentences.append(new_sentence)
        new_sentence = ['START']



#print(len(sentences))
#print(idx2word)
print("Word count ", len(idx2word))

word_count = Counter(all_words)



# RNNs are extrememly sensitive to outliers. So we
# create a list of all words only appearing once and
# remove them from the training set
unique_word_list = []
for k, v in word_count.items():
    if v == 1:
        unique_word_list.append(k)
#print(unique_word_list)


#####################################################################
# remove rare words from our sentences, word2idx, idx2word lists
# and redo our word lists, dictionarys, counts etc
# this reduces unique words down to 2411

words = data.split()
new_sentence = ['START']
sentences = []
word2idx = {'START':0, 'END':1}
idx2word = ['START', 'END']
current_idx = 2
all_words = []
max_sentence_length = 0


for w in words:

    # check for unique word and replace it with unique token
    try:
        index = unique_word_list.index(w)
        w = 'unique_word'

    except ValueError:
        # all is well continue on...
        print ()


    # split into sentences
    if w != 'END_SENTENCE':
        new_sentence.append(w)

        # add to word indexes if it's not already there
        if w not in idx2word:
            idx2word.append(w)
            word2idx.update({w : current_idx})
            current_idx += 1
        all_words.append(w)
    else:
        if len(new_sentence) > 6:       # start/end, drop one off x, one off y - remove small sentences
            new_sentence.append('END')
            sentences.append(new_sentence)
            if len(new_sentence) > max_sentence_length: 
                max_sentence_length = len(new_sentence)
        new_sentence = ['START']
        


        



#print(len(sentences))
#print(idx2word)
print("Word count ", len(idx2word))
word_count = Counter(all_words)
print(word_count)

print("Max sentence length (needed for word_embedding)", max_sentence_length)
print("Training sentences:", len(sentences))

# save dictionaries
with open('word2idx.pkl', "wb") as f:
    pickle.dump(word2idx, f)

np.save('sentences.npy', sentences)
np.save('idx2word.npy', idx2word)
np.save('unique_word_list.npy', unique_word_list)

######################################################
# test open files
tokenized_text = np.load('sentences.npy')
idx2word = np.load('idx2word.npy')
unique_word_list = np.load('unique_word_list.npy')
word2idx = pickle.load(open('word2idx.pkl', "rb"))


print("test opened files")
