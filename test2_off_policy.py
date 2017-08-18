import itertools
import argparse
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import collections
import gym
from envs import create_atari_env
import tensorflow as tf 

parser = argparse.ArgumentParser(description='A2C')
parser.add_argument('--num-agents', type=int, default=16)
parser.add_argument('--t-max', type=int, default=20)
parser.add_argument('--imsize', type=int, default=42)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--env-name', default='PongDeterministic-v4')

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
Transition_pr = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "prob"])
exp_rep = []
exp_rep_size = 10000
batch_size = 32
gamma = 0.99


class Estimator():
    def __init__(self, num_actions, lr, imsize, scope="estimator"):
        self.num_actions = num_actions
        self.lr = lr
        self.imsize = imsize
        self.scope = scope

        with tf.variable_scope(scope):
            self._build_model()

    def _build_model(self):
        self.states = tf.placeholder(shape=[None, self.imsize, self.imsize, 4], dtype=tf.float32, name='X')
        self.targets_pi = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_pi")
        self.targets_v = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_v") 
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        batch_size = tf.shape(self.states)[0]
        conv1 = tf.contrib.layers.conv2d(self.states, 32, 3, 2, activation_fn=tf.nn.relu, scope="conv1")
        conv2 = tf.contrib.layers.conv2d(conv1, 32, 3, 2, activation_fn=tf.nn.relu, scope="conv2")
        conv3 = tf.contrib.layers.conv2d(conv2, 32, 3, 2, activation_fn=tf.nn.relu, scope="conv3")
        conv4 = tf.contrib.layers.conv2d(conv3, 32, 3, 2, activation_fn=tf.nn.relu, scope="conv4")
        
        flattened = tf.contrib.layers.flatten(conv4)

        ### Policy
        fc1 = tf.contrib.layers.fully_connected(flattened, 256, activation_fn=tf.nn.relu)
        self.logits_pi = tf.contrib.layers.fully_connected(fc1, self.num_actions, activation_fn=None)
        self.probs_pi = tf.nn.softmax(self.logits_pi) + 1e-8

        self.entropy_pi = -tf.reduce_sum(self.probs_pi * tf.log(self.probs_pi), 1, name="entropy")

        gather_indices_pi = tf.range(batch_size) * self.num_actions + self.actions
        self.picked_action_probs_pi = tf.gather(tf.reshape(self.probs_pi, [-1]), gather_indices_pi)

        self.losses_pi = - (tf.log(self.picked_action_probs_pi) * self.targets_pi + 0.01 * self.entropy_pi)
        self.loss_pi = tf.reduce_sum(self.losses_pi, name="loss_pi")

        ### Value
        self.logits_v = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1, activation_fn=None)
        self.logits_v = tf.squeeze(self.logits_v, squeeze_dims=[1])

        self.losses_v = tf.squared_difference(self.logits_v, self.targets_v)
        self.loss_v = tf.reduce_sum(self.losses_v, name="loss_v")

        # Combine loss
        self.loss = self.loss_pi + 0.5*self.loss_v
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.train_op_v = tf.train.AdamOptimizer(self.lr).minimize(self.loss_v)


class Agent():
    def __init__(self, index, sess, env, model_net, t_max):
        self.name = 'agent-{}'.format(index)
        self.sess = sess
        self.env = env
        self.model_net = model_net
        self.t_max = t_max
        self.episode_counter = 1

        self._reset()

    def _reset(self):
        self.total_reward = 0
        self.episode_length = 0
        self.action_counter = [0] * self.model_net.num_actions

        state = self.env.reset() 
        self.state = np.stack([state] * 4, axis=2)

    def _policy_net_predict(self, state):
        feed_dict = { self.model_net.states: [state] }
        probs = self.sess.run(self.model_net.probs_pi, feed_dict)
        return probs[0] 

    def _value_net_predict(self, state):
        feed_dict = { self.model_net.states: [state] }
        logits_v = self.sess.run(self.model_net.logits_v, feed_dict)
        return logits_v[0]

    def run_n_steps(self):
        transitions = []
        for _ in range(self.t_max):
            action_probs = self._policy_net_predict(self.state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, rew, done, _ = self.env.step(action)
            reward = np.clip(rew, -1, 1)
            next_state = np.append(self.state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
           
            self.total_reward += rew
            self.episode_length += 1
            self.action_counter[action] += 1

            transitions.append(Transition(
                state=self.state, action=action, reward=reward, next_state=next_state, done=done))
            ### ugly
            exp_rep.append(Transition_pr(
                state=self.state, action=action, reward=reward, next_state=next_state, done=done, prob=action_probs[action]))
            if len(exp_rep) == exp_rep_size:
                exp_rep.pop(0)

            if len(exp_rep) >= exp_rep_size / 5:
                samples = random.sample(exp_rep, batch_size)
                states, actions, rewards, next_states, dones, probs_old = map(np.array, zip(*samples))

                probs_new = self.sess.run(self.model_net.probs_pi, {self.model_net.states: states})
                probs_new = probs_new[range(batch_size), actions]
                values_next = self.sess.run(self.model_net.logits_v, {self.model_net.states: next_states})

                values_target = probs_new / probs_old * (rewards + np.invert(dones).astype(np.float32) * gamma * values_next)

                sess.run(self.model_net.train_op_v, {
                    self.model_net.states: states,
                    self.model_net.targets_v: values_target,
                    self.model_net.actions: actions
                })
            ###

            if done:
                print("{}, episode {}, total_reward {}, episode_length {}, action distr {}".format(
                    self.name, self.episode_counter, self.total_reward, self.episode_length, self.action_counter))
                
                self._reset()
                self.episode_counter += 1
                break
            else:
                self.state = next_state
        return transitions


class Coordinator():
    def __init__(self, sess, num_agents, envs, model_net, gamma, t_max):
        self.sess = sess
        self.num_agents = num_agents
        self.envs = envs
        self.model_net = model_net
        self.gamma = gamma
        self.t_max = t_max
        self.agents = []
        self.num_updates = 0
        self.start_time = time.time()

        for i in range(num_agents):
            agent = Agent(index=i, sess=sess, env=envs[i], model_net=model, t_max=t_max)
            self.agents.append(agent)

    def _value_net_predict(self, state):
        feed_dict = { self.model_net.states: [state] }
        logits_v = self.sess.run(self.model_net.logits_v, feed_dict)
        return logits_v[0]

    def run(self):
        transitions_batch = []
        for i in range(self.num_agents):
            transitions = self.agents[i].run_n_steps()
            transitions_batch.append(transitions)

        shuf_arr = np.arange(0, self.num_agents)
        random.shuffle(shuf_arr)
        losses = []
        for i in shuf_arr:
            losses.append(self.update(transitions_batch[i]))
        
        self.num_updates += 1
        print('update {}, loss {}'.format(self.num_updates, np.mean(losses)))
        if self.num_updates % 1000 == 1:
            print('time {}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - self.start_time))))

    def update(self, transitions):
        reward = 0.0
        if not transitions[-1].done:
            reward = self._value_net_predict(transitions[-1].next_state)

        states = []
        actions = []
        policy_targets = []
        value_targets = []

        for transition in transitions[::-1]:
            reward = transition.reward + self.gamma * reward
            policy_target = (reward - self._value_net_predict(transition.state))

            states.append(transition.state)
            actions.append(transition.action)
            policy_targets.append(policy_target)
            value_targets.append(reward)

        feed_dict = {
            self.model_net.states: states,
            self.model_net.targets_pi: policy_targets,
            self.model_net.targets_v: value_targets,
            self.model_net.actions: actions,
        }

        mnet_loss, _ = self.sess.run([
            self.model_net.loss,
            self.model_net.train_op
        ], feed_dict)

        return mnet_loss


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)        
    tf.set_random_seed(args.seed)

    envs = []
    for rank in range(args.num_agents):
        env = create_atari_env(args.env_name)
        env.seed(args.seed + rank)
        envs.append(env)
    num_actions = envs[0].action_space.n

    model = Estimator(num_actions=num_actions, lr=args.lr, imsize=args.imsize)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        
        coord = Coordinator(sess=sess, num_agents=args.num_agents, envs=envs, model_net=model, gamma=args.gamma, t_max=args.t_max)

        while True:
            coord.run()
