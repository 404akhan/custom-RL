import gym
import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

from estimators import *

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def make_copy_params_op(v1_list, v2_list):
  v1_list = list(sorted(v1_list, key=lambda v: v.name))
  v2_list = list(sorted(v2_list, key=lambda v: v.name))

  update_ops = []
  for v1, v2 in zip(v1_list, v2_list):
    op = v2.assign(v1)
    update_ops.append(op)

  return update_ops

def make_train_op(local_estimator, global_estimator):
  """
  Creates an op that applies local estimator gradients
  to the global estimator.
  """
  local_grads, _ = zip(*local_estimator.grads_and_vars)
  _, global_vars = zip(*global_estimator.grads_and_vars)
  local_global_grads_and_vars = list(zip(local_grads, global_vars))
  return global_estimator.optimizer.apply_gradients(local_global_grads_and_vars,
          global_step=tf.contrib.framework.get_global_step())


class Worker(object):
  def __init__(self, name, env, model_net, global_counter, discount_factor=0.99, summary_writer=None, max_global_steps=None):
    self.name = name
    self.discount_factor = discount_factor
    self.max_global_steps = max_global_steps
    self.global_step = tf.contrib.framework.get_global_step()
    self.global_model_net = model_net
    self.global_counter = global_counter
    self.local_counter = itertools.count()
    self.summary_writer = summary_writer
    self.env = env

    # Create local policy/value nets that are not updated asynchronously
    with tf.variable_scope(name):
      self.model_net = Estimator(model_net.num_outputs)

    # Op to copy params from global policy/valuenets
    self.copy_params_op = make_copy_params_op(
      tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
      tf.contrib.slim.get_variables(scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES))

    self.mnet_train_op = make_train_op(self.model_net, self.global_model_net)

    self.state = None
    self.sum_reward = 0.
    self.current_life = 0
    
  def is_dead(self, info):
    dead = False
    if self.current_life > info['ale.lives']:
        dead = True
    self.current_life = info['ale.lives']
    return dead

  def run(self, sess, coord, t_max):
    with sess.as_default(), sess.graph.as_default():
      # Initial state
      self.state = atari_make_initial_state(self.env.reset())
      try:
        while not coord.should_stop():
          # Copy Parameters from the global networks
          sess.run(self.copy_params_op)

          # Collect some experience
          transitions, local_t, global_t = self.run_n_steps(t_max, sess)

          if self.max_global_steps is not None and global_t >= self.max_global_steps:
            tf.logging.info("Reached global step {}. Stopping.".format(global_t))
            coord.request_stop()
            return

          # Update the global networks
          gl_step, loss = self.update(transitions, sess)

      except tf.errors.CancelledError:
        return

  def _policy_net_predict(self, state, sess):
    feed_dict = { self.model_net.states: [state] }
    probs = sess.run(self.model_net.probs_pi, feed_dict)
    return probs[0] 

  def _value_net_predict(self, state, sess):
    feed_dict = { self.model_net.states: [state] }
    logits_v = sess.run(self.model_net.logits_v, feed_dict)
    return logits_v[0]

  def run_n_steps(self, n, sess):
    transitions = []
    for _ in range(n):
      # Take a step
      action_probs = self._policy_net_predict(self.state, sess)
      action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
      next_state, rew, done, info = self.env.step(action)
      reward = max(min(rew, 1), -1)
      dead = self.is_dead(info)
      next_state = atari_make_next_state(self.state, next_state)
      self.sum_reward += rew

      # Store transition
      transitions.append(Transition(
        state=self.state, action=action, reward=reward, next_state=next_state, done=done or dead))

      # Increase local and global counters
      local_t = next(self.local_counter)
      global_t = next(self.global_counter)

      if local_t % 100 == 0:
        tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))

      if done:
        self.state = atari_make_initial_state(self.env.reset())
        print(self.name, local_t, self.sum_reward)
        self.sum_reward = 0.
        break
      else:
        self.state = next_state

      if dead:
        self.state = atari_make_initial_state(next_state[:,:,-1])
        break

    return transitions, local_t, global_t

  def update(self, transitions, sess):
    # If we episode was not done we bootstrap the value from the last state
    reward = 0.0
    if not transitions[-1].done:
      reward = self._value_net_predict(transitions[-1].next_state, sess)

    # Accumulate minibatch exmaples
    states = []
    policy_targets = []
    value_targets = []
    actions = []

    for transition in transitions[::-1]:
      reward = transition.reward + self.discount_factor * reward
      policy_target = (reward - self._value_net_predict(transition.state, sess))
      # Accumulate updates
      states.append(transition.state)
      actions.append(transition.action)
      policy_targets.append(policy_target)
      value_targets.append(reward)

    feed_dict = {
      self.model_net.states: np.array(states),
      self.model_net.targets_pi: policy_targets,
      self.model_net.actions: actions,
      self.model_net.targets_v: value_targets,
    }

    # Train the global estimator using local gradients
    global_step, loss, _ = sess.run([
      self.global_step,
      self.model_net.loss,
      self.mnet_train_op
    ], feed_dict)

    return global_step, loss
