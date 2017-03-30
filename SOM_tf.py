

# http://github.com/timestocome


# Working through MEAP Machine Learning w/ TensorFlow Book
# added a few things to their sample code 


import tensorflow as tf 
import numpy as np



class SOM:

    def __init__(self, width, height, dim):
        
        self.num_iters = 100
        
        self.width = width
        self.height = height
        self.dim = dim 
        self.node_locs = self.get_locs()

        # 2d grid 
        nodes = tf.Variable(tf.random_normal([width * height, dim]))
        self.nodes = nodes

        # input data 
        x = tf.placeholder(tf.float32, [dim])
        iter = tf.placeholder(tf.float32)

        self.x = x
        self.iter = iter

        # get closest matching node 
        bmu_loc = self.get_bmu_loc(x)

        # update neighbor's values
        self.propagate_nodes = self.get_propagation(bmu_loc, x, iter)


        # update neighbor weights 
    def get_propagation(self, bmu_loc, x, iter):

        num_nodes = self.width * self.height 

        # reduce rate over time to settle weights
        rate = 1.0 - tf.div(iter, self.num_iters)
        alpha = rate * 0.5 
        sigma = rate * tf.to_float(tf.maximum(self.width, self.height)) / 2.
        
        # get closest matches, distances
        expanded_bmu_loc = tf.expand_dims(tf.to_float(bmu_loc), 0)
        sqr_dists_from_bmu = tf.reduce_sum(tf.square(tf.subtract(expanded_bmu_loc, self.node_locs)), 1)
        neigh_factor = tf.exp(-tf.div(sqr_dists_from_bmu, 2 * tf.square(sigma)))
        

        rate = tf.multiply(alpha, neigh_factor)
        rate_factor = tf.pack([tf.tile(tf.slice(rate, [i], [1]), [self.dim]) for i in range(num_nodes)])

        nodes_diff = tf.multiply(rate_factor, tf.subtract(tf.pack([x for i in range(num_nodes)]), self.nodes))
        update_nodes = tf.add(self.nodes, nodes_diff)

        return tf.assign(self.nodes, update_nodes)


        # get location of closest node
    def get_bmu_loc(self, x):

        expanded_x = tf.expand_dims(x, 0)
        sqr_diff = tf.square(tf.subtract(expanded_x, self.nodes))
        dists = tf.reduce_sum(sqr_diff, 1)
        bmu_idx = tf.argmin(dists, 0)
        bmu_loc = tf.pack([tf.mod(bmu_idx, self.width), tf.div(bmu_idx, self.width)])

        return bmu_loc


        # get locations of all nodes on the grid    
    def get_locs(self):

        # ? get very different som with x, y swapped
        locs = [[x, y] for y in range(self.height) for x in range(self.width)]
        return tf.to_float(locs)


    # run 
    def train(self, data):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(self.num_iters):
                for data_x in data:
                    sess.run(self.propagate_nodes, {self.x: data_x, self.iter: i})

            centroid_grid = [[] for i in range(self.width)]
            self.nodes_val = list(sess.run(self.nodes))
            self.locs_val = list(sess.run(self.node_locs))

            for i, l in enumerate(self.locs_val):
                centroid_grid[int(l[0])].append(self.nodes_val[i])
            self.centroid_grid = centroid_grid

###############################################################################
import matplotlib.pyplot as plt
import numpy as np




colors = np.array(
    [[0., 0., 1.],
    [0., 1, 0.05], 
    [1., 0., 0.], 
    [0., 1., 0.], 
    [1., 0.05, 0.], 
    [0., 0., 0.95], 
    [1., 0., 0.05], 
    [0., 0.95, 0.],
    [0., 0.05, 1.], 
    [1., 1., 0.]])


# train som 
som = SOM(4, 4, 3)
som.train(colors)


# plots
# bar plot to display input data
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1 = plt.bar(x, y, color=colors)
plt.title("Inputs")


ax2 = fig.add_subplot(212)
ax2 = plt.imshow(som.centroid_grid)
plt.title("SOM")


plt.show()       