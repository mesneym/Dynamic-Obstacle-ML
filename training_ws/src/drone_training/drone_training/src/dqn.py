# Inspired by https://keon.io/deep-q-learning/
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent():
    def __init__(self,action_size, state_size, gamma=1.0, alpha=0.01, epsilon=1.0, epsilon_log_decay = 0.9995,
                 epsilon_min=0.1, alpha_decay=0.01):

        self.action_size = action_size                                                #action space
        self.state_size = state_size                                                  #state space

        self.memory = deque(maxlen=100000)                                            #memory to store (s,r,s',a')

        self.gamma = gamma                                                            #discount factor

        self.epsilon = epsilon                                                        #Epsilon for epsilon greedy policy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay

        self.alpha = alpha                                                            #learning rate
        self.alpha_decay = alpha_decay

        self.model = Sequential()                                                     #NN model
        self.model.add(Dense(60, input_dim=self.state_size, activation='relu'))
        self.model.add(Dense(75, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        
        

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon): 
        return  random.randrange(self.action_size) if (np.random.random() <= self.epsilon) else np.argmax(self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state,(1,self.state_size))

    def replay(self,batch_size = 100):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



