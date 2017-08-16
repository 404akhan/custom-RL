import itertools
import argparse
import os
import time
import argparse
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import sys
import random
import collections
import gym

parser = argparse.ArgumentParser(description='A2C')
parser.add_argument('--num-agents', type=int, default=16)
parser.add_argument('--t-max', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.00025)
parser.add_argument('--env-name', default='PongDeterministic-v4')

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class StateProcessor():
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        return sess.run(self.output, { self.input_state: state })


class Estimator():
    def __init__(self, num_actions, lr, scope="estimator"):
        self.num_actions = num_actions
        self.lr = lr
        self.scope = scope

        with tf.variable_scope(scope):
            self._build_model()

    def _build_model(self):
        self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='X')
        self.targets_pi = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_pi")
        self.targets_v = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_v") 
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        batch_size = tf.shape(self.states)[0]
        conv1 = tf.contrib.layers.conv2d(self.states, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
        conv2 = tf.contrib.layers.conv2d(conv1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2")
        
        flattened = tf.contrib.layers.flatten(conv2)

        ### Policy
        fc1 = tf.contrib.layers.fully_connected(flattened, 256)
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
        self.loss = self.loss_pi + self.loss_v
        self.optimizer = tf.train.RMSPropOptimizer(self.lr, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss)


class Agent():
    def __init__(self, index, sess, env, state_pr, model_net, t_max):
        self.name = 'agent-{}'.format(index)
        self.sess = sess
        self.env = env
        self.state_pr = state_pr
        self.model_net = model_net
        self.t_max = t_max
        self.episode_counter = 1

        self._reset()

    def _reset(self):
        self.total_reward = 0
        self.episode_length = 0
        self.action_counter = [0] * self.model_net.num_actions

        state = self.env.reset() 
        state = self.state_pr.process(self.sess, state)
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
            next_state = self.state_pr.process(self.sess, next_state)
            reward = np.clip(rew, -1, 1)
            next_state = np.append(self.state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
           
            self.total_reward += rew
            self.episode_length += 1
            self.action_counter[action] += 1

            transitions.append(Transition(
                state=self.state, action=action, reward=reward, next_state=next_state, done=done))

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
    def __init__(self, sess, num_agents, envs, state_pr, model_net, gamma, t_max):
        self.sess = sess
        self.num_agents = num_agents
        self.envs = envs
        self.state_pr = state_pr
        self.model_net = model_net
        self.gamma = gamma
        self.t_max = t_max
        self.agents = []
        self.num_updates = 0
        self.start_time = time.time()

        for i in range(num_agents):
            agent = Agent(index=i, sess=sess, env=envs[i], state_pr=state_pr, model_net=model, t_max=t_max)
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
        loss = self.update(transitions_batch)
        
        self.num_updates += 1
        print('update {}, loss {}'.format(self.num_updates, loss))
        if self.num_updates % 1000 == 1:
            print('time {}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - self.start_time))))

    def update(self, transitions_batch):
        states = []
        actions = []
        policy_targets = []
        value_targets = []

        for i in range(self.num_agents):
            transitions = transitions_batch[i]

            reward = 0.0
            if not transitions[-1].done:
              reward = self._value_net_predict(transitions[-1].next_state)

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

        mnet_loss, _ = sess.run([
          self.model_net.loss,
          self.model_net.train_op
        ], feed_dict)

        return mnet_loss


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)        

    envs = []
    for _ in range(args.num_agents):
        env = gym.envs.make(args.env_name)
        envs.append(env)
    num_actions = envs[0].action_space.n

    state_pr = StateProcessor()
    model = Estimator(num_actions=num_actions, lr=args.lr)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        
        coord = Coordinator(sess=sess, num_agents=args.num_agents, envs=envs, state_pr=state_pr, model_net=model, gamma=args.gamma, t_max=args.t_max)

        while True:
            coord.run()
