import tensorflow as tf


class DNN(object):

    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden1 = 100
        self.n_hidden2 = 100
        self.learning_rate = 0.001

    def init_model(self):

        # Input
        self.X = tf.placeholder(tf.float32, shape=(None, self.n_inputs), name="X")
        self.y = tf.placeholder(tf.int64, shape=(None), name="Y")

        with tf.name_scope("dnn"):
            self.hidden1 = tf.layers.dense(self.X, self.n_hidden1, name="hidden1", activation=tf.nn.relu)
            self.hidden2 = tf.layers.dense(self.hidden1, self.n_hidden2, name="hidden2", activation=tf.nn.relu)
            self.logits = tf.layers.dense(self.hidden2, self.n_outputs, name="outputs")

    def get_training_operation(self):
        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(self.logits, self.Y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # training
        with tf.name_scope("train"):
            self.training = tf.train.GradientDescentOptimizer(self.learning_rate).maximize(accuracy)

        return self.training
