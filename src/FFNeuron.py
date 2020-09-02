import numpy
import math

MOMENTUM = 0.05


class FFNeuron:

    def __init__(self, index, in_list=None, in_len=0, activation_function='linear'):
        if in_list is None:
            in_list = []
        if in_len == 0:
            self.input_weights = in_list
        else:
            temp = numpy.random.uniform(low=-1.0, high=1.0, size=(in_len,))
            self.input_weights = list(temp)

        if activation_function == 'linear':
            self.activate = self.linear
        elif activation_function == 'sigmoid':
            self.activate = self.sigmoid
        else:
            self.activate = activation_function

        self.last_output = 0
        self.delta = 0
        self.weight_modifier = None
        self.last_input = []
        self.index = index

    def applyWeights(self, input_vector):
        """
        Support method which multiplies input vector by weights.
        :param input_vector: pandas Series. vector to be modified by the weights.
        :return: float.
        """
        result = 0
        for i in range(len(self.input_weights)):
            result += input_vector[i] * self.input_weights[i]
        return result

    def updateWeights(self):
        """
        Support method for training. adds the new weight modifier to the old.
        """
        self.input_weights = numpy.array(self.input_weights) + numpy.array(self.weight_modifier)

    def outputAdjust(self, expected, use_momentum=False, learning_rate=0.01):
        """
        Modifies the output layer values based on momentum, expected values, weight, and learning rate.
        :param expected: float. expected/actual value.
        :param use_momentum: Boolean. momentum yes or no.
        :param learning_rate: float. learning rate multiplier.
        """
        actual = self.last_output
        diff = expected - actual

        if self.activate == self.linear:
            self.delta = diff
        else:
            self.delta = diff * (actual * (1 - actual))

        if use_momentum and self.weight_modifier is not None:
            weight_modifier = (learning_rate * self.delta) * numpy.array(self.last_input)
            self.weight_modifier = weight_modifier + (MOMENTUM * numpy.array(self.weight_modifier))
        else:
            self.weight_modifier = (learning_rate * self.delta) * numpy.array(self.last_input)

    def hiddenAdjust(self, next_layer, use_momentum=False, learning_rate=0.01):
        """
        Modifies the output layer values based on momentum, weight, and learning rate.
        :param next_layer: next layer to pass the new values to.
        :param use_momentum: Boolean. momentum yes or no.
        :param learning_rate: float. learning rate multiplier.
        """
        output = self.last_output
        derivative = output * (1 - output)
        previous_layer_sum = 0

        for neuron in next_layer:
            previous_layer_sum += (neuron.delta * neuron.input_weights[self.index])
        self.delta = derivative * previous_layer_sum

        if use_momentum and self.weight_modifier is not None:
            weight_modifier = (learning_rate * self.delta) * numpy.array(self.last_input)
            self.weight_modifier = numpy.array(weight_modifier) + (MOMENTUM * numpy.array(self.weight_modifier))
        else:
            self.weight_modifier = (learning_rate * self.delta) * numpy.array(self.last_input)

    def sigmoid(self, input_vector):
        """
        Applies the sigmoid function to the input vector.
        :param input_vector: pandas Series.
        :return: float.
        """
        self.last_input = input_vector
        sig_input = self.applyWeights(input_vector)
        if sig_input < 0:
            power = math.exp(sig_input)
            output = 1 - (1 / (1 + power))
            self.last_output = output
            return output
        else:
            power = math.exp(-sig_input)
            output = 1/(1 + power)
            self.last_output = output
            return output

    def linear(self, input_vector):
        """
        propogates the input vector values multiplied by the weights.
        :param input_vector: pandas Series.
        :return: pandas Series.
        """
        self.last_input = input_vector
        result = self.applyWeights(input_vector)
        self.last_output = result
        return result
