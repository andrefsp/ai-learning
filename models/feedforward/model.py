import numpy as np
import tensorflow as tf


class FeedForwardConvTF(object):
    def __init__(self, image_shape=None):
        self.image_shape = image_shape
        self.shape = self.image_shape + (1,)

        self.x = tf.placeholder(
            shape=(None,) + self.shape, dtype='float32', name='image'
        )

        net = tf.layers.conv2d(
            self.x,
            10, (5, 5),
            strides=(2, 2),
            data_format="channels_last",
        )
        net = tf.layers.flatten(net)
        self.model = tf.layers.dense(net, 10, activation=tf.nn.sigmoid)

    def predict_single(self, obj):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            return session.run(
                self.model,
                feed_dict={
                    self.x: obj.astype(np.float32).reshape((1,) + self.shape)
                })

    def predict_many(self, obj):
        tf_input = obj.astype(np.float32).reshape((len(obj),) + self.shape)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            return session.run(
                self.model,
                feed_dict={
                    self.x: tf_input
                })


class FeedForwardConv(object):

    def __init__(self, image_shape=None):
        self.image_shape = image_shape
        self.shape = self.image_shape + (1,)

        self.model = tf.keras.Sequential()
        self.model.add(
            tf.layers.Conv2D(
                10, (5, 5),
                strides=(2, 2),
                data_format="channels_last",
                input_shape=self.shape
            )
        )
        self.model.add(tf.layers.Dense(128, activation=tf.nn.sigmoid))
        self.model.add(tf.layers.Flatten())
        self.model.add(tf.layers.Dense(10, activation=tf.nn.softmax))

    def predict_many(self, objs):
        return self.model.predict(
            objs.astype(np.float32).reshape((len(objs),) + self.shape)
        )

    def predict_single(self, obj):
        return self.model.predict(
            obj.astype(np.float32).reshape((1,) + self.shape)
        )


class FeedForward(object):

    def __init__(self, shape=None):
        self.shape = shape

        self.model = tf.keras.Sequential()
        self.model.add(tf.layers.Flatten(input_shape=self.shape))
        self.model.add(tf.layers.Dense(200, activation=tf.nn.relu))
        self.model.add(tf.layers.Dense(10, activation=tf.nn.softmax))
        self.optimizer = tf.keras.optimizers.Adam(lr=0.002)
        self.model.compile(self.optimizer, loss=tf.losses.mean_squared_error)

    def predict_many(self, objs):
        return self.model.predict(
            np.array(objs.astype(np.float32)).reshape(len(objs), 28, 28),
        )

    def predict_single(self, obj):
        return self.model.predict(
            np.array(obj.astype(np.float32)).reshape(1, 28, 28)
        )

    def train(self, x_train, y_train, x_validate, y_validate):
        to_categorical = tf.keras.utils.to_categorical

        self.model.fit(
            x=x_train,
            y=to_categorical(y_train),
            validation_data=(x_validate, to_categorical(y_validate)),
            epochs=10,
        )
