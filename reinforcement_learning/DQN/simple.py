#Deep learning stuff
import tensorflow as tf

#numerical computing in python
import numpy as np

#basic matlab-style plotting
import matplotlib.pyplot as plt

#see how fast our code is running
import time

#openAI gym, for are rl environments
import gym

#interactive plotting mode
plt.ion()

#helper function for creating nn layers
def linear(inp,out_dim,name,activation_fn=None,bias=True):
    in_dim = inp.get_shape()[1]
    with tf.variable_scope(name):
        W = tf.get_variable('W',[in_dim,out_dim],tf.float32)
        out = tf.matmul(inp,W)
        if bias:
            b = tf.get_variable('b',[out_dim],tf.float32)
            out = out + b
        if activation_fn != None:
            out = activation_fn(out)
    return out

#setup environment
#env = gym.make('MountainCar-v0')
#env = gym.make('FrozenLake-v0')
env = gym.make('CartPole-v0')
env.render()

#set hyper-parameters
learning_rate = 1e-5
gamma = .9
epsilon = .1
#assuming vector state-space (change for image-based environments)
in_dim = env.observation_space.shape[0] 
hid_dim = 100
#assuming discrete action space, otherwise... use other code
out_dim = env.action_space.n

#setup computational graph
x = tf.placeholder(tf.float32,shape=[1,in_dim])
target = tf.placeholder(tf.float32,shape=[1,1])
hid_layer = linear(x,hid_dim,'hidden',tf.nn.relu)
q = linear(hid_layer,out_dim,'q_values')
loss = tf.reduce_mean(tf.reduce_sum(tf.square(target-q),1))
optim = tf.train.AdamOptimizer(learning_rate)
train_step = optim.minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
s = env.reset()
#setup info variables
r_hist = []
q_hist = []
episode_count = 0
cumr = 0.0
cumq = 0.0
cumloss = 0.0
cur_time = time.clock()
refresh = int(1e2)
first_q = None


for t in range(int(1e6)):
    q_vals = sess.run(q,feed_dict={x:np.expand_dims(s,0)})
    if first_q is None:
        first_q = np.max(q_vals)
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_vals)
    #env.render()
    sPrime,r,done,_ = env.step(action)
    cumr+=r
    if done:
        cumq +=first_q
        first_q = None
        sPrime = env.reset()
        step_target = r
        episode_count+=1
    else:
        step_target = r + gamma*np.max(sess.run(q,feed_dict={x:np.expand_dims(sPrime,0)}))
    step_target = [[step_target]]
    _,step_loss = sess.run([train_step,loss],feed_dict={x:np.expand_dims(s,0),target:step_target})
    cumloss+=step_loss
    if (episode_count+1) % refresh ==  0:
        r_hist.append(cumr/refresh)
        q_hist.append(cumq/refresh)
        print(t,r_hist[-1],cumloss,time.clock()-cur_time)
        plt.clf()
        plt.plot(list(range(len(r_hist))),r_hist,list(range(len(q_hist))),q_hist)
        plt.pause(.1)
        episode_count = 0
        cumr = 0.0
        cumq = 0.0
        cumloss = 0.0
        cur_time = time.clock()

