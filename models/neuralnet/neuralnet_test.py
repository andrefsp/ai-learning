import unittest
from unittest import TestCase
from neuralnet import NeuralNet
from neuralnet import Layer


class LayerTestCase(TestCase):

    def test_layer(self):
        layer = Layer(input_size=4, size=2)
        self.assertEquals(len(layer.weighs), 8)
        self.assertEquals(len(layer.nodes), 2)

    def test_predict(self):

        def multiply(vector):
            value = 1
            for i in vector:
                value *= i
            return value

        layer = Layer(input_size=3, size=2, activation_function=sum)
        output = layer.predict([1, 1, 1])
        self.assertEquals(output, [3, 3])


class NeuralNetTestCase(TestCase):

    def test_predict(self):
        neural_net = NeuralNet(
            input_size=3,
            hidden_size=3,
            output_size=1,
        )

        result = neural_net.predict([1, 1, 1])
        self.assertEquals(result, 6)

unittest.main()
