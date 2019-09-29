# import numpy as np
import tensorflow as tf
from .preprocess import ALPHABET
from .preprocess import preprocess


class LSTM(object):

    def __init__(self, input_size, alphabet=ALPHABET):
        self.input_size = input_size
        self.alphabet = alphabet

        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.InputLayer(input_shape=(self.input_size,))
        )
        self.model.add(
            tf.keras.layers.Embedding(
                len(self.alphabet),
                100,
                input_length=self.input_size
            )
        )
        self.model.add(tf.keras.layers.LSTM(200))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    def predict(self, strings):
        return self.model.predict(
            preprocess(strings, input_length=self.input_size)
        )
