

class Node(object):

    def __init__(self, function):
        self.function = function

    @property
    def activation_function(self):
        return self.function

    def calculate(self, input_vector):
        return self.activation_function(input_vector)


class Layer(object):

    def __init__(self, input_size, size=None, activation_function=None):
        self.nodes = [Node(activation_function) for _ in range(size)]
        self.weighs = [1 for _ in range(size * input_size)]

    def predict(self, input_vector):
        weighed_input = []

        batch = len(self.nodes)

        for i, value in enumerate(input_vector):
            input_weigh = self.weighs[i:i + batch]
            for weigh in input_weigh:
                weighed_input.append(value * weigh)

        node_inputs = {}
        index = 0
        for value in weighed_input:
            if index >= batch:
                index = 0
            try:
                node_inputs[index].append(value)
            except KeyError:
                node_inputs[index] = [value, ]
            index += 1

        return [
            self.nodes[index].calculate(node_input)
            for index, node_input in node_inputs.items()
        ]


class NeuralNet(object):

    def __init__(self, input_size, hidden_layers):
        pass

    def predict(self, input):
        return 0
