from envs import create_atari_env
import unittest
import gym
import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing
from gym import wrappers

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

from estimators import *
from policy_monitor import PolicyMonitor
from worker import Worker


tf.flags.DEFINE_string("ckpt_dir", "./models/checkpoints-breakout", "Directory where checkpoint is saved.")
tf.flags.DEFINE_string("env", "Pong-v0", "Name of gym Atari environment, e.g. Pong-v0")
tf.flags.DEFINE_integer("t_max", 20, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 60, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", None, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS

def make_env(wrap=True):
  env = create_atari_env(FLAGS.env)
  return env

# Depending on the game we may have a limited action space
env = make_env()
env = wrappers.Monitor(env, './tmp-videos/run1')
VALID_ACTIONS = list(range(env.action_space.n))

CHECKPOINT_DIR = FLAGS.ckpt_dir

if not os.path.exists(CHECKPOINT_DIR):
  os.makedirs(CHECKPOINT_DIR)

def policy_net_predict(model_net, state, sess):
  feed_dict = { model_net.states: [state] }
  probs = sess.run(model_net.probs_pi, feed_dict)
  return probs[0] 


with tf.device("/cpu:0"):

  # Keeps track of the number of updates we've performed
  global_step = tf.Variable(0, name="global_step", trainable=False)

  # Global policy and value nets
  with tf.variable_scope("global") as vs:
    model_net = Estimator(num_outputs=len(VALID_ACTIONS))

  # Global step iterator
  global_counter = itertools.count()

  saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0, max_to_keep=10)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  # Load a previous checkpoint if it exists
  latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
  if latest_checkpoint:
    print("Loading model checkpoint: {}".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)

  for i_episode in range(110):
    state = atari_make_initial_state(env.reset())
    sum_reward = 0
    for t in itertools.count():
      action_probs = policy_net_predict(model_net, state, sess)
      action = np.argmax(action_probs)
      next_state, reward, done, info = env.step(action)
      sum_reward += reward
      state = atari_make_next_state(state, next_state)

      if done:
        print("Episode finished after {} timesteps, {} reward".format(t+1, sum_reward))
        sum_reward = 0
        break