import tensorflow as tf
import numpy as np

learning_rate = 0.001

X_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

Y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

model_save_path = './saved/linear.cckp'

export_path = './serve/'


class LinearModel(object):

    def __init__(self):
        # Inputs
        self.X = tf.placeholder(tf.float32, name='X')
        self.Y = tf.placeholder(tf.float32, name='Y')

        # Variables to optimize
        self.M = tf.Variable(np.random.rand(), name='M')
        self.B = tf.Variable(np.random.rand(), name='B')

        # model:  Y = M*X + B
        self.model = tf.add(tf.multiply(self.M, self.X), self.B)

        # Model save and serving setup.
        self.builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        self.tensor_info_X = tf.saved_model.utils.build_tensor_info(self.X)
        self.tensor_info_model = tf.saved_model.utils.build_tensor_info(self.model)

        self.signature_def_map = {
            'serving_default': (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'X': self.tensor_info_X},
                    outputs={'Y': self.tensor_info_model},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            ),
        }

    def get_training_operation(self):

        # Mean squared error
        # Compare the output of the model with training 'Y'
        error = tf.reduce_sum(tf.pow(self.model - self.Y, 2) / len(2 * X_train))

        # Minimize the Mean Squared Error with an Adam Optimizer
        training = tf.train.AdamOptimizer(learning_rate).minimize(error)
        return training

    def get_train_data_batches(self):
        # Make sure data can be feed in batches in case a too big dataset.
        for i in range(100):
            yield {self.X: X_train, self.Y: Y_train}

    def train(self):
        training = self.get_training_operation()

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            print("start:\t Y = %s*X + %s" % (session.run(self.M), session.run(self.B)))

            for epoch in range(100):
                for train_batch in self.get_train_data_batches():
                    session.run(training, feed_dict=train_batch)

            print("end:\tY = %s*X + %s" % (session.run(self.M), session.run(self.B)))

            # Save the model
            self.builder.add_meta_graph_and_variables(
                session,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map=self.signature_def_map,
            )

            self.builder.save()

    def predict(self, X):

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            # Initialize session
            session.run(init)

            # Load saved model
            tf.saved_model.loader.load(
                session, [
                    tf.saved_model.tag_constants.SERVING
                ],
                export_path
            )

            # predict
            return session.run(self.model, feed_dict={self.X: X})


linear_model = LinearModel()

linear_model.train()

for x in range(4, 10):
    print("Prediction: M * %s + B = %s" % (x, linear_model.predict(x)))
