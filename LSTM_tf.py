

# http://github.com/timestocome

# Working through MEAP Machine Learning w/ TensorFlow Book
# added a few things to their sample code and fixed a bug



# uses airline data  ( I removed top line / headings )
# https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line



import tensorflow as tf
import numpy as np



class SeriesPredictor:

    def __init__(self, n_input, window_size, n_hidden=24):

        self.n_input = n_input
        self.window_size = window_size
        self.n_hidden = n_hidden

        self.W_out = tf.Variable(tf.random_normal([n_hidden, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, window_size, n_input])
        self.y = tf.placeholder(tf.float32, [None, window_size])

        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        self.saver = tf.train.Saver()



    def model(self):

        cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        n_samples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [n_samples, 1, 1])

        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)

        return out 


    def train(self, train_x, train_y, test_x, test_y):

        with tf.Session() as sess:

            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())

            max_patience = 2
            patience = max_patience
            min_test_err = float('inf')
            step = 0
            
            while patience > 0:
                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
            
                if step % 100 == 0:
            
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    print('step: {}\t\ttrain err: {}\t\ttest err: {}'.format(step, train_err, test_err))
            
                    if test_err < min_test_err:
                        min_test_err = test_err
                        patience = max_patience
                    else:
                        patience -= 1
                step += 1
            
            save_path = self.saver.save(sess, 'model.ckpt')
            print('Model saved to {}'.format(save_path))


    def test(self, sess, test_x):

        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, './model.ckpt')
        output = sess.run(self.model(), {self.x: test_x})
        
        return output

#######################################################################

import pandas as pd
import matplotlib.pyplot as plt



def load_series(filename, idx=1):

    z = pd.read_csv(filename)
    z.columns = ['date', 'volume']
    data = z['volume'].tolist()

    normalized_data = (data - np.mean(data)) / np.std(data)

    return normalized_data        


def split_data(data, percent_train= 0.80):

    n_rows = len(data)
    train_data, test_data = [], []


    for idx, row in enumerate(data):
        if idx < n_rows * percent_train:
            train_data.append(row)
        else:
            test_data.append(row)

    return train_data, test_data

    



def plot_results(train_x, predictions, actual, filename):

    plt.figure()
    num_train = len(train_x)
    
    plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
    plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
    plt.legend()
    
    if filename is not None:
        plt.savefig(filename)
    plt.show()


window_size = 12        # yearly cycle, one data point per month
predictor = SeriesPredictor(n_input=1, window_size=window_size, n_hidden=24)

data = load_series('international-airline-passengers.csv')
train_data, test_data = split_data(data)

train_x, train_y = [], []
for i in range(len(train_data) - window_size - 1):
    train_x.append(np.expand_dims(train_data[i:i+window_size], axis=1).tolist())
    train_y.append(train_data[i+1:i+window_size+1])

test_x, test_y = [], []
for i in range(len(test_data) - window_size -1):
    test_x.append(np.expand_dims(test_data[i:i+window_size], axis=1).tolist())
    test_y.append(test_data[i+1:i+window_size+1])




predictor.train(train_x, train_y, test_x, test_y)


with tf.Session() as sess:

    predicted_values = predictor.test(sess, test_x)[:, 0]
    print("predictions:", np.shape(predicted_values))
    plot_results(train_data, predicted_values, test_data, 'Predictions_train_test.png')

    previous_sequence = train_x[-1]
    predicted_values = []



    for i in range(20):

        next_sequence = predictor.test(sess, [previous_sequence])
        predicted_values.append(next_sequence[-1])
        previous_sequence = np.vstack((previous_sequence[1:], next_sequence[-1]))

    plot_results(train_data, predicted_values, test_data, 'Predictions_network.png')


#######################################################################

'''
# test model
predictor = SeriesPredictor(n_input=1, window_size=4, n_hidden=10)

train_x = [[[1], [2], [5], [6]],
            [[5], [7], [7], [8]],
            [[3], [4], [5], [7]]] 

train_y = [[1, 3, 7, 11],
               [5, 12, 14, 15],      
                [3, 7, 9, 12]] 
                
predictor.train(train_x, train_y)


test_x = [[[1], [2], [3], [4]], [[4], [5], [6], [7]]]
predictor.test(test_x)
'''