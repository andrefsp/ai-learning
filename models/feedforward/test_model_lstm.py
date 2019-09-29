import unittest
from .model_lstm import LSTM


class LSTMTestCase(unittest.TestCase):

    def test_lstm_prediction_single(self):
        model = LSTM(1000)
        prediction = model.predict('andre')
        self.assertEqual(len(prediction), 1)

    def test_lstm_prediction_many(self):
        model = LSTM(1000)
        prediction = model.predict(['andre', 'filipe'])
        self.assertEqual(len(prediction), 2)
