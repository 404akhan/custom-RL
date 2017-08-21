import numpy as np
import tensorflow as tf


def atari_make_initial_state(state):
  return np.stack([state] * 4, axis=2)

def atari_make_next_state(state, next_state):
  return np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

  
class Estimator():
    def __init__(self, num_outputs, scope="estimator"):
        self.num_outputs = num_outputs
        self.scope = scope

        with tf.variable_scope(scope):
            self._build_model()

    def _build_model(self):
        self.states = tf.placeholder(shape=[None, 42, 42, 4], dtype=tf.float32, name='X')
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
        self.logits_pi = tf.contrib.layers.fully_connected(fc1, self.num_outputs, activation_fn=None)
        self.probs_pi = tf.nn.softmax(self.logits_pi) + 1e-8

        self.entropy_pi = -tf.reduce_sum(self.probs_pi * tf.log(self.probs_pi), 1, name="entropy")

        gather_indices_pi = tf.range(batch_size) * self.num_outputs + self.actions
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

        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
          global_step=tf.contrib.framework.get_global_step())

