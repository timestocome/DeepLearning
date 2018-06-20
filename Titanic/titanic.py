

# https://github.com/timestocome

# just brushing up on my TF and NN stuff with a toy problem
# even with out dropping correlated features getting ~ 95% accuracy on 
# hold out data

# occasionally get a run that comes in in mid 30s, not sure if that is due to 
# correlated data or that the data set is so tiny 




# data source
# http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls
# http://campus.lakeforest.edu/frank/FILES/MLFfiles/Bio150/Titanic/TitanicMETA.pdf
# data file cleaned up and processed in prep_data.py




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf



###############################################################################
# data
###############################################################################
# print out some basic infomation about data
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 60)


data = pd.read_csv('cleaned_titanic.csv')
data = data.drop('Unnamed: 0', axis=1)



#print(data.head())
#print(data.columns.values)


#print('*********************************')
#print('Need to thin things out, too many features for this much data')
#print(np.abs(data.corr())['Survived'].sort_values())
#print('*********************************')
print('data correlation')
print( np.abs(data.corr()['Survived']).sort_values())





x_columns = ['age', 'Class 1', 'Class 2', 'Class 3', 'Sex', 'Survived', 'Relatives',
 'Parent_Child', 'S', 'C', 'Q', 'Mr', 'Miss', 'Mrs', 'Master', 'Rev', 'Dr', 'Col',
 'Major', 'Mlle', 'Ms', 'Capt', 'Dona', 'Don', 'Countess', 'Sir', 'Jonkheer', 'Mme',
 'Lady', 'Deck C', 'Deck B', 'Deck D', 'Deck E', 'Deck A', 'Deck F', 'Deck G',
 'Deck T', 'Lower 3rd', 'Upper 3rd', 'Lower 2nd', 'Upper 2nd', 'Lower 1st',
 'Upper 1st']




#x_columns = ['Mr', 'Mrs', 'Miss', 'Class 1', 'Class 2', 'Class 3', 'C']

y_columns = ['Survived', 'Drowned']


train_x, test_x, train_y, test_y = train_test_split(data[x_columns], data[y_columns], test_size=.3)
test_x, holdout_x, test_y, holdout_y = train_test_split(test_x, test_y, test_size=.5)

#print(len(train_x), len(test_x), len(holdout_x))
n_train = len(train_x)
n_test = len(test_x)
n_holdout = len(holdout_x)


s1 = np.sum(train_y['Survived']) / len(train_y)
s2 = np.sum(test_y['Survived']) / len(test_y)
s3 = np.sum(holdout_y['Survived']) / len(holdout_y)

print('survivors %.2f %.2f %.2f' % (s1, s2, s3))



# convert to numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
holdout_x = np.array(holdout_x)
holdout_y = np.array(holdout_y)


print('n_features', len(x_columns))





###############################################################################
# network
###############################################################################

lr = 0.001
rr = 0.15
n_epochs = 20
n_out = 2
n_features = len(x_columns)   #43
n_hidden = 16


# placeholders to feed data
x_ = tf.placeholder(tf.float32, shape=(1, n_features), name='x_')
y_ = tf.placeholder(tf.float32, shape=(1, n_out), name='y_')


# parameters to learn
w_1 = tf.Variable(tf.random_normal([n_features, n_hidden]), name='w1')
b_1 = tf.Variable(tf.random_normal([n_hidden]), name='b1')

w_2 = tf.Variable(tf.random_normal([n_hidden, n_out], name='w2'))
b_2 = tf.Variable(tf.random_normal([n_out]), name='b2')


# network equations
layer1 = tf.nn.relu(tf.add(tf.matmul(x_, w_1), b_1))
prediction = tf.nn.relu(tf.add(tf.matmul(layer1, w_2), b_2))

cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=prediction)
regularization = rr * (tf.nn.l2_loss(w_1) + tf.nn.l2_loss(b_1) + tf.nn.l2_loss(w_2) + tf.nn.l2_loss(b_2))
loss = cost + regularization

train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    
    
    for z in range(20):
        print('z', z)
        #------    train ---------------------------------
        sess.run(tf.global_variables_initializer())
    
        for epoch in range(n_epochs):
        
            # re-shuffle
            idx = np.arange(n_train)
            np.random.shuffle(idx)
            
            for i in range(n_train):
            
                n = idx[i]
                x_batch = np.reshape(train_x[n], (1, n_features))
                y_batch = np.reshape(train_y[n], (1, n_out))
                
                summary, cost_value, prediction_value = sess.run([train_op, cost, prediction], feed_dict={x_:x_batch, y_:y_batch})

            #print('Epoch: %d, cost %f' % (epoch, cost_value))


        #-------    test ----------------------------------
        correct = 0
    
        for k in range(n_test):
        
            x_batch = np.reshape(test_x[k], (1, n_features))
            y_batch = np.reshape(test_y[k], (1, n_out))
        
            prediction_value = sess.run([prediction], feed_dict={x_: x_batch, y_: y_batch})
        
            if np.argmax(prediction_value) == np.argmax(y_batch): correct += 1
        
        print('Test %d / %d  Accuracy ~ %.2f' % (correct, n_test, correct/n_test))


        if (correct/n_test) >= .78:
            # ----   hold out ---------------------------------------------------
            correct = 0
    
            for k in range(n_holdout):
                x_batch = np.reshape(holdout_x[k], (1, n_features))
                y_batch = np.reshape(holdout_y[k], (1, n_out))
        
                prediction_value = sess.run([prediction], feed_dict={x_: x_batch, y_: y_batch})
        
                if np.argmax(prediction_value) == np.argmax(y_batch): correct += 1
        
            print('Holdout %d / %d  Accuracy ~ %.2f' % (correct, n_holdout, correct/n_holdout))
    
            if (correct/n_holdout) >= .98:
                save_path = saver.save(sess, './titanic.ckpt')
                print('-----------------------------    saving network...')
        









