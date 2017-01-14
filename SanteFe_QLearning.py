
# http://github.com/timestocome


###########################################################
# to do 
# check maze and q_expected updating each loop
# need visuals to see if maze route gets better with training
###########################################
import numpy as np

import os
import sys
import timeit

import theano
import theano.tensor as T



# setup theano
GPU = True
if GPU:
    print("Device set to GPU")
    try: theano.config.device = 'gpu'
    except: pass    # its already set
    theano.config.floatX = 'float32'
else:
    print("Running with CPU")


# tuning values -- no reason to keep passing constants into a function
n_epochs = 10            # number of games to play
n_moves = 512           # max moves per game
batch_size = 1          # refeed updated maze back in after each move

L1_reg = 0.0            # possibly multiple solutions/unstable
L2_reg = 0.0001         # provides one stable solution
learning_rate = 0.01
epsilon = 0.30          # how often to make a random move
discount = .99          # discounted rewards

n_hidden = 48
n_pixels = 32 * 32      # maze size 
n_actions = 4           # possible actions (0-3) up, right, down, left


rng = np.random.RandomState(27)     # prime random number generator

# for plotting, visuals etc
all_paths = np.zeros((n_epochs, n_moves))
all_rewards = np.zeros((n_epochs, n_moves))

####################################################################################
# set up data
####################################################################################

# the SantaFe Ant Trail Maze
new_maze = np.asarray([
        1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 
], dtype='float32')


# init
current_maze = theano.shared(np.asarray(new_maze, dtype=theano.config.floatX), name='current_maze', borrow=True)
q_expected = theano.shared(np.asarray(rng.uniform(low=0, high=1., size=(n_pixels, n_actions)), dtype=theano.config.floatX), name='q_expepected', borrow=True)



###############################################################################################
# input to hidden layer
#################################################################################################
class HiddenLayer(object):

    def __init__(self, input=current_maze, in_in=n_pixels, n_out=n_hidden, W=None, b=None):
        
        if W is None:
            W_values = np.asarray(rng.uniform(
                low = -np.sqrt(2./(n_pixels + n_out)),
                high = np.sqrt(2./(n_pixels + n_out)),
                size = (n_pixels, n_out)
            ), dtype=theano.config.floatX)
        W = theano.shared(value=W_values, name='W', borrow=True)
        self.W = W

        if b is None:
            b_values = np.zeros((n_out, ), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
        self.b = b

        self.params = [self.W, self.b]
        self.output = T.nnet.relu(T.dot(input, self.W) + self.b)
        self.input = input      # current maze



##################################################################################
# Q Layer, outputs best action to take in current state
###################################################################################
class QLayer(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):

        # Q maps rewards onto each of the 4 actions possible per location
        # this tracks expected reward based on previous experience
        if W is None:
            W_values = np.asarray(rng.uniform(low=0, high=1., size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)
        self.W = W

        if b is None:
            b = theano.shared(value=np.zeros((n_out), dtype=theano.config.floatX), name='b', borrow=True)
        self.b = b

        # vars 
        self.params = [self.W, self.b]
        self.input = input        

        self.location = 0                       # start in location zero
        self.action = -1
        self.rewards = 1                     # first location has a reward
        
    
        # predicted action and greedy action to take
        self.predicted_action_rewards = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.greedy_action = T.argmax(self.predicted_action_rewards)


    def new_game(self):
        self.location = 0
        self.action = -1
        self.rewards = 1


    # (actual reward - expected)^2 
    def cost(self):
        return -T.mean(self.predicted_action_rewards - q_expected[self.rewards][self.action] **2)



    # bellman equations => expected return discounted over time - actual reward
    def update_q(self):

        # update reward actions matrix
        reward_mask = np.zeros((n_pixels, n_actions)).astype('float32')    # mask 
        reward_mask[self.location][self.action] = self.rewards 

        # bellman equation update
        q_mask = T.matrix('q_maxtrix')                   # placeholder in graph
        update_q = theano.function([q_mask], q_expected, updates = {q_expected: q_expected * discount - q_mask})
        update_q(np.array(reward_mask, dtype=theano.config.floatX))


    # move ant
    def move(self):

        state = self.location

        # random actior or greedy?
        action_to_try = -1
        if np.random.rand(1) <= epsilon:  action_to_try = np.random.randint(0,3)   # random
        else: action_to_try = self.greedy_action.eval()                            # network picks action to take
        
        
        # take a step if we can with out falling off edge of our world
        if action_to_try == 0:         # up
            if state > 31:                                  state = state - 32
        elif action_to_try == 1:       # right
            if ((state + 1) % 32 != 0 and state < 1023):    state = state + 1 
        elif action_to_try == 2:       # down 
            if state < (1023-31):                           state = state + 32
        elif action_to_try == 3:       # left
            if (state % 32 != 0 and state > 0):             state = state - 1
        

        # check for reward and update vars
        self.action = action_to_try
        self.predicted_reward = self.predicted_action_rewards[0][self.action]
        self.reward = int((current_maze[state]).eval())    # get reward if any 
        self.rewards += self.reward

        # update current location in maze
        self.location = state

        # if reward > 0 remove it from the maze
        # no need to update if the game board hasn't changed
        # updating shared data is a challenge in theano

        if self.reward > 0:

            # mask to subtract one from reward location
            delete_reward = np.zeros(n_pixels).astype('float32')
            delete_reward[self.location] = 1

            # perform update on maze
            update_maze = theano.function([], current_maze, updates=[(current_maze, current_maze - delete_reward)])
            update_maze()
    
        return self.location, self.action, self.reward
        
        


###########################################################################################
# define the network
#######################################################################################
class QNetwork(object):

    def __init__(self, input, n_in, n_hidden, n_out):

         # create layers
        self.hiddenLayer = HiddenLayer(input=input)
        self.qLayer = QLayer(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_actions)


        # regularization to prevent over training
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.qLayer.W).sum()
        self.L2 = (self.hiddenLayer.W **2).sum() + (self.qLayer.W **2).sum()
        self.params = self.hiddenLayer.params + self.qLayer.params


        # input data
        self.input = input

        # current ant location in maze
        #self.location = self.qLayer.location
        #self.rewards = self.qLayer.rewards
        #self.action = self.qLayer.action

        # methods for updating progress
        self.cost = self.qLayer.cost
        self.update_q = self.qLayer.update_q
        self.move = self.qLayer.move
        self.new_game = self.qLayer.new_game



def train_network(current_maze):

    # init network
    q_network = QNetwork(current_maze, n_in=n_pixels, n_out=n_actions, n_hidden=n_hidden)
    qLayer = q_network.qLayer
   

    # update network weights, bias
    cost = q_network.cost() + L1_reg * q_network.L1 + L2_reg * q_network.L2
    d_params = [T.grad(cost, param) for param in q_network.params]
    update_parameters = [ (param, param - learning_rate * d_param) for param, d_param in zip(q_network.params, d_params)]

    calculate_cost = theano.function(inputs=[], outputs=cost, updates=update_parameters)

    
    

    # train the model
    start_time = timeit.default_timer()
    epoch = 0
    avg_cost = 999999
    track_paths = []
    track_rewards = []

    print(".................begin training........")
    for epoch in range(n_epochs):
        print("New game, resetting maze")

        current_maze = theano.shared(np.asarray(new_maze, dtype=theano.config.floatX), borrow=True)
        q_network.new_game()


        track_path = [0,]
        track_reward = 1

        for move in range(n_moves):
            state, action, reward = q_network.move()
            cost = q_network.cost()
            q_update = q_network.update_q()

            track_path.append(state)
            track_reward += reward
            #print("track_reward, reward", track_reward, reward)

        print("cost", cost.eval())
        track_paths.append(track_path)
        track_rewards.append(track_reward)

    
    print("rewards", track_rewards)
    print("paths through maze")
    for p in track_paths: print(p)




train_network(current_maze)
