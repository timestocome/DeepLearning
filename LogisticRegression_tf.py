
# http://github.com/timestocome

# Working through MEAP Machine Learning w/ TensorFlow Book
# added a few things to their sample code 


# create some sample data and cluster using logistic regression 


import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 



# constants
learning_rate = 0.01
training_epochs = 1000
batch_size = 100




# create training data
x1_label0 = np.random.normal(1, 1, (100, 1))
x2_label0 = np.random.normal(1, 1, (100, 1))

x1_label1 = np.random.normal(5, 1, (100, 1))
x2_label1 = np.random.normal(4, 1, (100, 1))

x1_label2 = np.random.normal(8, 1, (100, 1))
x2_label2 = np.random.normal(0, 1, (100, 1))


xs_label0 = np.hstack((x1_label0, x2_label0))
xs_label1 = np.hstack((x1_label1, x2_label1))
xs_label2 = np.hstack((x1_label2, x2_label2))
xs = np.vstack((xs_label0, xs_label1, xs_label2))


# one hot vectors for output
labels = np.matrix( [[1., 0., 0.]] * len(x1_label0) +
                    [[0., 1., 0.]] * len(x1_label1) +
                    [[0., 0., 1.,]] * len(x1_label2))



# shuffle data
arr = np.arange(xs.shape[0])
np.random.shuffle(arr)
xs = xs[arr, :]
labels = labels[arr, :]


# get number of samples, features and labels 
train_size, num_features = xs.shape
num_labels = labels.shape[1]



# create test data
test_x1_label0 = np.random.normal(1, 1, (10, 1))
test_x2_label0 = np.random.normal(1, 1, (10, 1)) 
test_x1_label1 = np.random.normal(5, 1, (10, 1))
test_x2_label1 = np.random.normal(4, 1, (10, 1))
test_x1_label2 = np.random.normal(8, 1, (10, 1))
test_x2_label2 = np.random.normal(0, 1, (10, 1))
test_xs_label0 = np.hstack((test_x1_label0, test_x2_label0))
test_xs_label1 = np.hstack((test_x1_label1, test_x2_label1))
test_xs_label2 = np.hstack((test_x1_label2, test_x2_label2))
test_xs = np.vstack((test_xs_label0, test_xs_label1, test_xs_label2))
test_labels = np.matrix( [[1., 0., 0.]] * 10 + [[0., 1., 0.]] * 10 + [[0., 0., 1.]] * 10)




# display training data
plt.scatter(x1_label0, x2_label0, c='r', marker='o', s=60)
plt.scatter(x1_label1, x2_label1, c='g', marker='o', s=60)
plt.scatter(x1_label2, x2_label2, c='b', marker='o', s=60)

plt.scatter(test_x1_label0, test_x2_label0, c='r', marker='+', s=240)
plt.scatter(test_x1_label1, test_x2_label1, c='g', marker='+', s=240)
plt.scatter(test_x1_label2, test_x2_label2, c='b', marker='+', s=240)
plt.show()



# model
X = tf.placeholder('float', shape=[None, num_features])
Y = tf.placeholder('float', shape=[None, num_labels])

W = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))
y_model = tf.nn.softmax(tf.matmul(X, W) + b)

cost = -tf.reduce_sum(Y * tf.log(y_model))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))




# run model
with tf.Session() as sess:

    tf.global_variables_initializer().run()

    for step in range(training_epochs * train_size // batch_size):
        offset = (step * batch_size) % train_size
        batch_xs = xs[offset : (offset + batch_size), :]
        batch_labels = labels[offset : (offset + batch_size), :]

        err, _ = sess.run([cost, train_op], {X: batch_xs, Y: batch_labels})


    W_val = sess.run(W)
    print('w', W_val)
    
    b_val = sess.run(b)
    print('b', b_val)

    print('accuracy', accuracy.eval({X: test_xs, Y: test_labels}))


    # get predicted values for test points
    predictions = y_model.eval({X: test_xs, Y: test_labels })
    prediction_values = tf.argmax(predictions, 1)
    p = prediction_values.eval()
    t = np.argmax(test_labels, 1)
    for i, j in zip(p, t): print(i, j)

    

sess.close()