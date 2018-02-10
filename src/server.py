from keras import models

import quoridor_env
import quoridor
import tensorflow as tf
from keras import backend as K
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, LeakyReLU, Flatten, AveragePooling1D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import plot_model
from gym.spaces import prng
import os.path

GPU = True
CPU = False
num_cores = 4

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
                        inter_op_parallelism_threads=num_cores, allow_soft_placement=True, \
                        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

class QuoridorModel:
    def __init__(self, env, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=4096, minibatches_per_episode=5, quiet=False):
        self.memory = deque(maxlen=100000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.quiet = quiet
        self.action_count = env.action_space.n
        self.minibatches_per_episode = minibatches_per_episode
        self.sx = env.unwrapped.sx
        self.sy = env.unwrapped.sy

        sz = self.sx*self.sy
        self.model = Sequential()
        self.model.add(Conv2D(sz, kernel_size=3, input_shape=(self.sx, self.sy, 5), data_format='channels_last'))
        for i in range(2):
            self.model.add(Conv2D(sz, kernel_size=3))
            self.model.add(LeakyReLU(0.1))

        self.model.add(MaxPooling2D())
        print(self.model.output_shape)
        self.model.add(Flatten())

        self.model.add(Dense(self.sx, activation='linear'))
        self.model.add(Dense(self.sx * 2, activation='linear'))
        self.model.add(Dense(self.sx * 4, activation='linear'))
        self.model.add(Dense(self.sx * 8, activation='linear'))
        self.model.add(Dense(self.sx * 16, activation='linear'))
        self.model.add(Dense(self.sx * 16, activation='tanh'))
        self.model.add(Dense(env.action_space.n, activation='linear'))
        self.model.add(Dense(env.action_space.n, activation='softmax'))
        self.model.add(Dense(env.action_space.n, activation='softmax'))
        self.model.add(Dense(env.action_space.n, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def load(self, fname):
        self.model = load_model(fname)

    def dump(self, fname):
        self.model.save(fname)

    def preprocess(self, state):
        board = np.full((self.sx, self.sy, 5), 0)
        nplayers = 2
        offset = nplayers * 2 + 1
        state = state[0]
        for x in range(self.sx):
            for y in range(self.sy):
                p = x * self.sy + y
                board[x][y][0] = 1 if state[1] == p else 0
                board[x][y][1] = 1 if state[2] == p else 0
                board[x][y][2] = 1 if int(state[offset + p]) & quoridor.DIR_RIGHT else 0
                board[x][y][3] = 1 if int(state[offset + p]) & quoridor.DIR_DOWN else 0
                board[x][y][4] = 1 if int(state[offset + p]) & quoridor.DIR_RIGHT_AND_DOWN else 0
        return np.array([board])

    def random_move(self, state):
        pid = state[0][0]
        player_cnt = 2
        if state[0][1 + player_cnt + int(pid)] == 0:
            return random.sample([0, 1, 2, 3], 1)[0]
        return prng.np_random.randint(self.action_count)

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t/10 + 1) * self.epsilon_decay)))

    def choose_action(self, state, episode):
        return self.random_move(state) if (np.random.random() <= self.get_epsilon(episode)) else np.argmax(self.model.predict(self.preprocess(state)))

    def process_result(self, state, action, reward, next_state, done):
        self.remember(self.preprocess(state), action, reward, self.preprocess(next_state), done)

    def replay(self, batch_size):
        #print("replay")
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))

        # we have very sparse rewards, so trying this to propagate it faster
        # minibatch += random.sample(
        #    self.positive_memory, min(len(self.positive_memory), self.positive_batch_injection))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    def train(self):
        for i in range(self.minibatches_per_episode):
            self.replay(self.batch_size)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class D2Solver():
    def __init__(self, n_episodes=101, n_win_ticks=195, max_env_steps=None, monitor=False, quiet=False):
        self.env = gym.make('quoridor-v0')
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.quiet = quiet
        self.print_every_turn = True
        self.nplayers = self.env.unwrapped.number_of_players
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.models = []
        for i in range(self.nplayers):
            m = QuoridorModel(self.env)
            fname = 'models/episode_%d_latest.bin' % i
            if os.path.isfile(fname):
                m.load(fname)
            self.models.append(m)
            m.dump('models/episode_%d_0.bin' % i)

    def preprocess_state(self, state):
        return np.reshape(state, [1, self.env.observation_space.n])

    def run(self):
        scores = deque(maxlen=100)
        episodes = range(5, 1000, 5)
        for num_of_episodes in episodes:
            for e in range(num_of_episodes):
                state = self.preprocess_state(self.env.reset())
                done = False
                total_rewards = np.full(self.nplayers, 0)
                turns = 0
                first_player = self.env.unwrapped.player
                while not done and turns < 1000:
                    pid = self.env.unwrapped.player
                    print('Turn %d for player %d' % (turns, pid))
                    action = self.models[pid].choose_action(state, e)
                    next_state, reward, done, _ = self.env.step(action)
                    if self.print_every_turn:
                        print(self.env.render(mode='ansi'))
                    next_state = self.preprocess_state(next_state)
                    self.models[pid].process_result(state, action, reward, next_state, done)
                    state = next_state
                    total_rewards[pid] += reward
                    turns += 1

                print(self.env.render(mode='ansi'))
                scores.append(total_rewards)
                mean_scores = np.mean(scores, 0)
                for pid in range(self.nplayers):
                    print('[Episode {}][Player {} Win? {} First? {}] - Mean score for last 100 {}'.format(e, pid, 'Y' if total_rewards[pid] == 1 else 'N', 'Y' if pid == first_player else 'N', mean_scores[pid]))
                    self.models[pid].train()
                    self.models[pid].dump('models/episode_%d_%d.bin' % (pid, e))
            self.models[pid].dump('models/episode_%d_latest.bin' % pid)


if __name__ == '__main__':
    agent = D2Solver()
    agent.run()