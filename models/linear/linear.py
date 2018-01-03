import tensorflow as tf
import numpy as np


learning_rate = 0.001


default_job_dir = './serve/'


class LinearModel(object):

    def __init__(self, job_dir=None):
        self.job_dir = job_dir or default_job_dir

    def init_model(self):
        # Inputs
        self.X = tf.placeholder(tf.float32, name='X')
        self.Y = tf.placeholder(tf.float32, name='Y')

        # Variables to optimize
        self.M = tf.Variable(np.random.rand(), name='M')
        self.B = tf.Variable(np.random.rand(), name='B')

        # model:  Y = M*X + B
        self.model = tf.add(tf.multiply(self.M, self.X), self.B, name='model')

        # Model save and serving signature map
        self.tensor_info_X = tf.saved_model.utils.build_tensor_info(self.X)
        self.tensor_info_model = tf.saved_model.utils.build_tensor_info(self.model)
        self.signature_def_map = {
            'serving_default': (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'X': self.tensor_info_X},
                    outputs={'model': self.tensor_info_model},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            ),
        }

    def get_training_operation(self):
        # Mean squared error
        # Compare the output of the model with training 'Y'
        # Root Squared error  (RSE) ??
        error = tf.reduce_sum(tf.pow(self.model - self.Y, 2))

        # Minimize the Mean Squared Error with an Adam Optimizer
        training = tf.train.AdamOptimizer(learning_rate).minimize(error)
        return training

    def train_from_file(self, train_file):
        # init file input
        filename_queue = tf.train.string_input_producer([train_file, ], shuffle=True)
        reader = tf.TextLineReader(skip_header_lines=1)
        _, rows = reader.read(filename_queue)

        lines = tf.decode_csv(rows, record_defaults=[[], []])
        X, Y = tf.train.shuffle_batch(lines, 10, 20, 10)

        # initiate variables for training
        self.init_model()

        # save builder
        builder = tf.saved_model.builder.SavedModelBuilder(self.job_dir)

        # get training operation
        training = self.get_training_operation()

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord)

            session.run(init)

            print("start:\t Y = %s*X + %s" % (session.run(self.M), session.run(self.B)))
            for epoch in range(10000):
                print("\t Epoch %s ::: \tY = %s*X + %s" % (epoch, session.run(self.M), session.run(self.B)))
                x_train, y_train = session.run([X, Y])

                session.run(training, feed_dict={self.X: x_train, self.Y: y_train})

            # Save the model
            builder.add_meta_graph_and_variables(
                session,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map=self.signature_def_map,
            )

            builder.save()

    def init_model_from_save(self, session):
        # Load saved model
        tf.saved_model.loader.load(
            session, [
                tf.saved_model.tag_constants.SERVING
            ],
            self.job_dir
        )

        graph = tf.get_default_graph()

        # Inputs
        self.Y = graph.get_tensor_by_name("Y:0")
        self.X = graph.get_tensor_by_name("X:0")

        # variables to optimise
        self.M = graph.get_tensor_by_name("M:0")
        self.B = graph.get_tensor_by_name("B:0")

        # model
        self.model = graph.get_tensor_by_name("model:0")

    def predict(self, X):

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            # Initialize session
            session.run(init)

            # initiate the model for prediction
            self.init_model_from_save(session)

            # predict
            return session.run(self.model, feed_dict={self.X: X})


# linear_model = LinearModel()

# linear_model.train_from_file()

# for x in range(4, 10):
#    print("Prediction: M * %s + B = %s" % (x, linear_model.predict(x)))
