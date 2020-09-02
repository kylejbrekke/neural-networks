from FFNeuron import FFNeuron
import numpy


class FFNetwork:
	def __init__(self, input_count, output_layer, dataset, hidden_layers=None, class_header="Class", learning_rate=0.1, verbose=False, regression=False, use_momentum=False):
		classes = numpy.unique(dataset[class_header])
		self.regression = regression
		self.use_momentum = use_momentum
		if verbose:
			if not regression:
				print("CLASSES IN DATASET:")
				for category in classes:
					print(category, end='  ')
				print()
			else:
				print("NO CLASSES IN REGRESSION DATASET")

		self.output_count = output_layer[0]
		print("\n%d NEURONS IN OUTPUT LAYER" % self.output_count) if verbose else None

		self.network = {'hidden': [], 'output': []}

		self.class_header = class_header
		self.learning_rate = learning_rate
		print("\nSELECTED LEARNING RATE: %f" % self.learning_rate) if verbose else None

		print("\nGENERATING EXPECTED VALUE TABLE:") if verbose else None
		expected_value_table = {}
		for i in range(len(classes)):
			category = classes[i]
			expected_value = []
			for j in range(len(classes)):
				if j == i:
					expected_value.append(1)
				else:
					expected_value.append(0)
			expected_value_table[category] = expected_value
			expected_value_table[str(expected_value)] = category
			print("Category: %s\tBinary: %s" % (category, str(expected_value))) if verbose else None
		self.expected_values = expected_value_table

		if hidden_layers is None:
			print("\nGENERATING OUTPUT LAYER", end='') if verbose else None
			layer = []
			for i in range(output_layer[0]):
				layer.append(FFNeuron(i, in_len=input_count, activation_function=output_layer[1]))
				print('.', end='') if verbose else None
			self.network['output'] = layer
			print("Output Layer Generated") if verbose else None
		else:
			for i in range(len(hidden_layers)):
				print("\nGENERATING HIDDEN LAYER %d" % (i + 1), end='') if verbose else None
				layer = []
				for j in range(hidden_layers[i][0]):
					if i == 0:
						layer.append(FFNeuron(i, in_len=input_count, activation_function=hidden_layers[i][1]))
					else:
						layer.append(FFNeuron(i, in_len=hidden_layers[i - 1][0], activation_function=hidden_layers[i][1]))
					print('.', end='') if verbose else None
				print() if verbose else None
				self.network['hidden'].append(layer)
			print("Hidden Layers Generated") if verbose else None

			print("\nGENERATING OUTPUT LAYER", end='') if verbose else None
			layer = []
			output_weight_count = len(self.network['hidden'][len(self.network['hidden']) - 1])
			for i in range(output_layer[0]):
				layer.append(FFNeuron(i, in_len=output_weight_count, activation_function=output_layer[1]))
				print('.', end='') if verbose else None
			self.network['output'] = layer
			print("Output Layer Generated") if verbose else None

	def classify(self, row, actual, verbose=False):
		"""
		Guesses a classification based on neural net.
		:rtype: object. If classification data set returns a tuple with classification guess, actual classification
						If regression data set returns classification guess. probably a float.
		"""
		if not self.regression:
			classification_probabilities = self.forwardPropagation(row)
			maximum = 0
			index = 0
			for i in range(len(classification_probabilities)):
				if classification_probabilities[i] > maximum:
					maximum = classification_probabilities[i]
					index = i

			classification_binaries = []
			for i in range(len(classification_probabilities)):
				if i != index:
					classification_binaries.append(0)
				else:
					classification_binaries.append(1)
			classification = self.expected_values[str(classification_binaries)]

			if verbose:
				print("CLASSIFICATION: %s\nACTUAL: %s\n" % (classification, actual))
			else:
				print("CLASSIFICATION: %s\tACTUAL: %s" % (classification, actual))
			return tuple([classification, actual])
		else:
			classification = (self.forwardPropagation(row))[0]
			if 1e-8 > classification:
				classification = 0
			if verbose:
				print("CLASSIFICATION: %f\nACTUAL: %f\nERROR: %f\n" % (classification, actual, actual - classification))
			else:
				print("CLASSIFICATION: %f\tACTUAL: %f" % (classification, actual))
			return classification

	def train(self, row):
		"""
		Trains hidden layer neurons and output neurons, runs back propogation.
		:rtype: object
		"""
		if not self.regression:
			expected = self.expected_values[row[self.class_header]]
		else:
			expected = float(row[self.class_header])

		vec = (row.drop(self.class_header)).values
		self.backPropagation(vec, expected)

	def backPropagation(self, row, expected):
		"""
		Performs back propogation on a row.
		:rtype: object
		"""
		previous_results = None
		loop = 1
		loop_limit = 20
		converged = False

		while loop <= loop_limit and not converged:
			actual_results = self.forwardPropagation(row)

			output_layer = self.network['output']
			if not self.regression:
				for i in range(len(output_layer)):
					output_layer[i].outputAdjust(expected[i], self.use_momentum, self.learning_rate)
			else:
				output_layer[0].outputAdjust(expected, self.use_momentum, self.learning_rate)
			next_layer = output_layer

			for layer in reversed(self.network['hidden']):
				for neuron in layer:
					neuron.hiddenAdjust(next_layer, self.use_momentum, self.learning_rate)
					next_layer = layer

			for neuron in self.network['output']:
				neuron.updateWeights()
			for layer in reversed(self.network['hidden']):
				for neuron in layer:
					neuron.updateWeights()

			loop += 1
			converged = True
			if previous_results is not None:
				for i in range(len(previous_results)):
					if ("%.5f" % previous_results[i]) != ("%.5f" % actual_results[i]):
						converged = False
						break
				previous_results = actual_results
			else:
				previous_results = actual_results
				converged = False

	def forwardPropagation(self, row):
		"""
		support method which performs forward propogation on a row. Going through every layer in the network and every
		neuron in the layer, we find the activation values given the row.
		:param row: pandas series. To be modified using activation values.
		:return: pandas Series. returns the freshly activated row.
		"""
		for layer in self.network['hidden']:
			output = []
			for neuron in layer:
				output.append(neuron.activate(row))
			row = output

		output = []
		for neuron in self.network['output']:
			output.append(neuron.activate(row))
		row = output

		return row
