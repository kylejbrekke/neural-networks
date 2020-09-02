import numpy
import sys
from Cluster import Cluster
from NearestNeighbor import NearestNeighbor
from FFNetwork import FFNetwork
from RBFNeuron import RBFNeuron


class RBFNetwork:
	def __init__(self, training_set, cluster_count, classes, set_reduction_method="means", verbose=False, regression=False, use_momentum=False, learning_rate=0.1):
		self.use_momentum = use_momentum
		self.training_set = training_set
		self.cluster_count = cluster_count
		self.class_header = classes.name
		self.unique_classes = numpy.unique(classes.values)
		self.regression = regression
		if verbose:
			if not regression:
				print("CLASSES IN DATASET:")
				for category in self.unique_classes:
					print(category, end='  ')
				print()
			else:
				print("NO CLASSES IN REGRESSION DATASET")

		self.learning_rate = learning_rate
		print("\nSELECTED LEARNING RATE: %f" % self.learning_rate) if verbose else None

		if not regression:
			self.output_count = len(self.unique_classes)
		else:
			self.output_count = 1

		self.set_reduction_method = str.lower(set_reduction_method)
		print("\nSELECTED REDUCTION METHOD FOR TRAINING SET: %s" % self.set_reduction_method) if verbose else None

		self.function_layer = []
		self.output_matrix = []
		self.output_layer = None
		self.prototypes = None
		print("\nGENERATING NEURONS") if verbose else None
		self.createNeurons(verbose)

	def createNeurons(self, verbose=False):
		"""
		Method which dictates calculating prototypes and sigma based on set_reduction_method.
		Also creates neurons and runs other methods for calculations regarding neurons.
		"""
		# Depending on the set_reduction_method, we use different algorithms to calculate prototypes
		if self.set_reduction_method == "means":
			print("Calculating centers for Gaussian function by means...")
			self.prototypes = Cluster.byMeans(self.training_set, number_of_clusters=self.cluster_count,
											  class_header=self.class_header, verbosity=0)
		elif self.set_reduction_method == "medoids":
			print("Calculating centers for Gaussian function by medoids...")
			self.prototypes = Cluster.byMedoids(self.training_set, self.cluster_count, self.class_header, verbosity=0)

		elif self.set_reduction_method == "condensed":
			print("Calculating centers for Gaussian function using condensed nearest neighbor...")
			self.prototypes = NearestNeighbor.condensedNearestNeighbor(self.training_set, self.class_header)

		else:
			print("'%s' is an invalid set reduction method, please check it and try again." % self.set_reduction_method)
			sys.exit()

		if not self.regression:
			print("Generating output layer of size %d with sigmoid activation functions..." % self.output_count) if verbose else None
			self.output_layer = FFNetwork(len(self.prototypes),
										  [self.output_count, 'sigmoid'],
										  self.training_set,
										  class_header=self.class_header,
										  learning_rate=self.learning_rate,
										  use_momentum=self.use_momentum,
										  regression=self.regression)
		else:
			print("Generating output layer with a single linear activation function for regression...") if verbose else None
			self.output_layer = FFNetwork(len(self.prototypes),
										  [self.output_count, 'linear'],
										  self.training_set,
										  class_header=self.class_header,
										  learning_rate=self.learning_rate,
										  use_momentum=self.use_momentum,
										  regression=self.regression)

		print("Generating widths for basis functions using nearest neighbor proximity...") if verbose else None
		sigma_list = self.findSigma()

		# for every point in prototype list, create a neuron and store that point and sigma in said neuron
		print("Generating layer of Gaussian basis functions of size %d..." % len(self.prototypes)) if verbose else None
		for i in range(len(self.prototypes)):
			self.function_layer.append(RBFNeuron(self.prototypes.iloc[i], sigma_list[i], self.class_header))

		print("\nTRAINING NEURONS ON TRAINING DATA OF %d ENTRIES" % len(self.training_set)) if verbose else None
		self.training_set.apply(lambda row: self.train(row), axis=1)

	def classify(self, row):
		"""
		Performs classification based on the input row.
		:rtype: modified output layer with new values.
		"""
		gauss_vector = self.applyGaussianLayer(row)
		expected = row[self.class_header]
		return self.output_layer.classify(gauss_vector, expected)

	def train(self, row):
		"""
		Support method which finds the output values for each neuron.
		"""
		gauss_vector = self.applyGaussianLayer(row)

		if not self.regression:  # not regression
			expected = self.output_layer.expected_values[row[self.class_header]]  # class value
		else:  # regression
			expected = float(row[self.class_header])  # float value

		self.output_layer.backPropagation(gauss_vector, expected)  # train the output layer

	def applyGaussianLayer(self, row):
		"""
		Has each neuron assess input row using the gaussian distribution.
		:param row: pandas Series.
		:return: List with new values for every row.
		"""
		gauss_vector = []
		for neuron in self.function_layer:
			gauss_vector.append(neuron.assessSimilarity(row))
		return gauss_vector

	def findSigma(self):
		"""
		Support method to find the sigma value by summing the distances from every point to its nearest neighbor,
		and dividing by the total number of points. This results in the same sigma for every prototype node.

		:return: List. distances calculated using 1NN.
		"""
		distance = []
		for index, row in self.prototypes.iterrows():
			modified_prototype_set = self.prototypes.drop([index])  # Remove current point from data set
			distance.append(NearestNeighbor.oneNearestNeighbor(row, modified_prototype_set, return_distance=True, class_header=self.class_header))

		return distance

