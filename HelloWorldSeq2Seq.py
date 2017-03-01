

# http://github.com/timestocome


# based on Google's sequence to sequence model
# https://www.tensorflow.org/tutorials/seq2seq

# lots of kludges in here, and it takes a lot of nodes, even for a simple example.

import tensorflow as tf
import numpy as np
import random




source_vocab_size = 128                     # only use ascii to make this simpler
target_vocab_size = 128                     # same size input as output ( in this case)
learning_rate = 0.3
max_in = 8
max_out = 8
buckets = [(max_in, max_out)]               # our input and response words can be up to max_in/max_out
batch_size = 12                             # how many samples between gradient adjustment of weights
number_of_nodes = 64                        # hidden nodes
number_of_layers = 2                        # hidden layers
dtype = tf.float32                          # keep from zeroing everthing out
padding = 3                                 # max_out - actual output length 

input_data = [104, 101, 108, 108, 111, 0, 0, 0] * batch_size            # hello in ascii plus padding
input_data = np.array(input_data)                                       # convert to numpy array
input_data = input_data.reshape(max_in, batch_size)                     # reshape

target_data = [119, 111, 114, 108, 100, 0, 0, 0] * batch_size           # world in ascii plus padding
target_data = np.array(target_data)
target_data = target_data.reshape(max_out, batch_size)

target_weights = np.ones((max_out, batch_size))                         # output mask
target_weights[5] = 0                                                   # zero out rows that are padded in target data
target_weights[6] = 0
target_weights[7] = 0



class HelloWorldSeq2Seq(object):

    def __init__(self):

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)

        cell = single_cell = tf.nn.rnn_cell.GRUCell(number_of_nodes)

        if number_of_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * number_of_layers)

		# The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
        
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, cell,
                    num_encoder_symbols = source_vocab_size,
                    num_decoder_symbols = target_vocab_size,
                    embedding_size = number_of_nodes,
                    feed_previous = do_decode)

		# Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        # setup placeholds to hold data we'll feed through network
        for i in range(max_in):	
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        
        for i in range(max_out + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))


        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

        self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                                            self.encoder_inputs, self.decoder_inputs, targets,
                                            self.target_weights, buckets,
                                            lambda x, y: seq2seq_f(x, y, False))


		# Gradients update operation for training the model.
        params = tf.trainable_variables()
        self.updates = []
        self.updates.append(tf.train.AdagradOptimizer(learning_rate).minimize(self.losses[0]))

        

    # loop over each letter in our input / output
    def step(self, session, encoder_inputs, decoder_inputs, test):

        bucket_id = 0 
        encoder_size = max_in
        decoder_size = max_out
        
        
		# Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

        for l in range(decoder_size ):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]


		# Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([batch_size], dtype=np.int32)


		# Output feed: depends on whether we do a backward step or not.
        if not test:
            output_feed = [self.updates[bucket_id], self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]      	# Loss for this batch.
            for l in range(decoder_size):	                # Output logits.
                output_feed.append(self.outputs[bucket_id][l])


        outputs = session.run(output_feed, input_feed)


        # training or testing?
        if not test: return outputs[0], outputs[1]              # Gradient norm, loss
        else:  return outputs[0], outputs[1:]                   # loss, outputs.




# translate ints to chars and string 'em together
def decode(bytes):
    
    last_row = bytes[-1]
    decoded_output = "".join(map(chr, last_row)).replace('\x00', '').replace('\n', '')

    return decoded_output[:(max_out-padding)]       # chomp off end beyond padded area


# are we there yet?
def test_progress():

    cost, outputs = model.step(session, input_data, target_data, test=True)
    
    words = np.argmax(outputs, axis=2)              # shape (inputs, outputs, vocab size)
    reply = decode(words)
    print("step %d, cost %f, output: %s " % (step, cost, reply))

    if cost < 0.15:   exit()                        # bail we're there or adjust if not
    

###############################################################################
# run code 

step = 0
test_step = 10
with tf.Session() as session:

    model = HelloWorldSeq2Seq()
    session.run(tf.global_variables_initializer())

    while True:
        model.step(session, input_data, target_data, test=False) # no outputs in training
        if step % test_step == 0:  test_progress()
        step = step + 1



# https://github.com/tensorflow/tensorflow/issues/3388
del session


