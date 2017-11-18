import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# env = gym.make('CartPole-v1')
env = gym.make('CartPole-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# print env.observation_space
# print env.action_space

class DQN:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.model = self._build_model()
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

	### Q-network, input is current state, return worth of an action
	def _build_model(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=0.001))
		return model

	def act(self, state):
		state = state.reshape(1, -1)
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		return np.argmax(self.model.predict(state)[0])

	def train(self, episodes=100):
		for i in range(episodes):
			## collecting more training data
			time = 0
			state = env.reset()
			done = False
			while not done:
				action = self.act(state)
				next_state, reward, done, _ = env.step(action)
				reward = reward if not done else -10
				time += 1
				self.memory.append((state, action, reward, next_state, done))
				state = next_state

			self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
			print "episode: %d, reward: %d, %4f" % (i, time, self.epsilon)

			## training
			if len(self.memory) < 100:
				continue
			minibatch = random.sample(self.memory, 32)
			for state, action, reward, next_state, done in minibatch:
				target = reward
				next_state = next_state.reshape(1, -1)
				state = state.reshape(1, -1)
				if not done:
					## r + \gamma * argmax(Q(s_t+1, a))
					# print self.model.predict(next_state)
					target = (reward + 0.95 * np.amax(self.model.predict(next_state)[0]))
				# set target state values
				target_f = self.model.predict(state)
				target_f[0][action] = target
				self.model.fit(state, target_f, epochs=1, verbose=0)


a = DQN(state_size, action_size)

a.train(1000)

state = env.reset()
done = False
while not done:
	env.render()
	action = a.act(state)
	next_state, reward, done, _ = env.step(action)
	state = next_state


