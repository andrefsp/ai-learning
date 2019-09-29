import unittest
import tensorflow as tf
from .model import FeedForward
from .model import FeedForwardConv
from .model import FeedForwardConvTF

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


class Mixin(object):

    model_class = None

    def test_predict_single(self):
        model = self.model_class(x_train[0].shape)
        prediction = model.predict_single(x_train[0])
        self.assertEqual(prediction.shape, (1, 10))
        for val in prediction[0]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_predict_many(self):
        model = self.model_class(x_train[0].shape)
        prediction = model.predict_many(x_train[:2])
        self.assertEqual(prediction.shape, (2, 10))


class FeedForwardConvTFTestCase(unittest.TestCase, Mixin):

    model_class = FeedForwardConvTF


class FeedForwardConvTestCase(unittest.TestCase, Mixin):

    model_class = FeedForwardConv


class FeedForwardTestCase(unittest.TestCase, Mixin):

    model_class = FeedForward

    def test_train_model(self):
        model = self.model_class(x_train[0].shape)
        model.train(x_train, y_train, x_test, y_test)
        prediction = model.predict_single(x_train[0])
        self.assertIsNotNone(prediction)
