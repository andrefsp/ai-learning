import tensorflow as tf


class DNN(object):

    def __init__(self, n_inputs, n_outputs):

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden1 = 100
        self.n_hidden2 = 100
        self.learning_rate = 0.001

        # Input
        self.x = tf.placeholder(tf.float32, shape=(None, self.n_inputs), name="X")
        self.y = tf.placeholder(tf.int64, shape=(None, self.n_outputs), name="Y")

        with tf.name_scope("dnn"):
            self.hidden1 = tf.layers.dense(self.x, self.n_hidden1, name="hidden1", activation=tf.nn.relu)
            self.hidden2 = tf.layers.dense(self.hidden1, self.n_hidden2, name="hidden2", activation=tf.nn.relu)
            self.logits = tf.layers.dense(self.hidden2, self.n_outputs, name="outputs")

        # Evaluation
        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(self.logits, self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # training
        with tf.name_scope("train"):
            self.training = tf.train.GradientDescentOptimizer(self.learning_rate).maximize(self.accuracy)
