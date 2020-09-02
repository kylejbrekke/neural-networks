import math


class RBFNeuron:

    def __init__(self, prototype, sigma, class_header):
        self.prototype = prototype
        self.sigma = sigma
        self.class_header = class_header

    def assessSimilarity(self, input_vector):
        """
        Uses the Gaussian function to find the similarity between the input vector and the prototype
        :param input_vector: Vector with the same length as prototype.
        """
        # activation function
        gauss = math.exp(-(((self.calcEuclideanDistance(input_vector)) ** 2) / (self.sigma**2)))
        if gauss < 1e-8:
            gauss = 0
        return gauss

    def calcEuclideanDistance(self, input_vector):
        """
        Support method which finds the Euclidean distance between the input vector and the prototype.
        :param input_vector: Series.
        :return: float. Euclidean Distance
        """
        distance = 0
        input_vector = input_vector.drop(self.class_header)
        prototype = self.prototype.drop(self.class_header)
        for i in range(len(input_vector)):  # for every value in the input vector
            distance += (input_vector[i] - prototype[i])**2  # add the distance from the prototype squared

        return math.sqrt(distance)
