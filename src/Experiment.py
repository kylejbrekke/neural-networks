import pandas
import numpy
import os
import math
from FFNetwork import FFNetwork
from RBFNetwork import RBFNetwork


class Experiment:
	def __init__(self, target, class_header="Class"):
		self.class_header = class_header
		target_parts = target.split('/')
		self.data_name = target_parts[len(target_parts) - 1]
		path = os.path.join(os.path.dirname(__file__), target)
		self.data = pandas.read_csv(path)
		self.input_count = len(self.data.columns) - 1
		self.classes = numpy.unique(self.data[class_header].values)
		self.class_index = self.data.columns.get_loc(self.class_header)
		print("_ _ _ _ _ _ _ _ _ _ New experimentation object created for dataset %s _ _ _ _ _ _ _ _ _ _\n" % self.data_name)

	def split(self, subset_count, regression=False):
		"""
		Separates the current experimental data (self.data) into subset_count number of datasets. The datasets have a
		proportional amount of each category in the master dataset (or as proportional an amount as possible). The
		classes are determined by evaluating the category_id column of the master dataset.

		:param subset_count: the number of datasets which self.data will be split into (integer)
		:return a list of distinct subsets from self.data stored as pandas.DataFrame objects
		"""
		print("Constructing %d subsets from %s" % (subset_count, self.data_name))
		subsets = [pandas.DataFrame(columns=self.data.columns)] * subset_count
		if not regression:
			for category in self.data[self.class_header].unique():  # for each category in the data
				print("\tPartitioning the %s category..." % category)
				category_dataset = self.data.loc[self.data[self.class_header] == category]
				proportion_size = math.ceil(len(category_dataset) / subset_count)
				proportions = [proportion_size] * subset_count

				print("\t\tModifying subsets to ensure proper size...")
				for i in range(subset_count):  # ensure the size of combined subsets is equal to the size of the whole data
					if numpy.sum(proportions) > len(category_dataset):
						proportions[i] -= 1
					else:
						break

				print("\t\tThe total size is: ", len(category_dataset))
				print("\t\tFinal subset sizes are: ", proportions)
				for i in range(subset_count):  # repeat for every subset
					sample = category_dataset.sample(n=proportions[i], replace=False)
					# remove sampled entries from the dataset
					category_dataset = category_dataset.drop(sample.index)
					subsets[i] = subsets[i].append(sample)  # Insert sampled entries into their corresponding subset
		else:
			data = self.data
			proportions = [math.ceil(len(self.data) / subset_count)] * subset_count

			for i in range(subset_count):  # ensure the size of combined subsets is equal to the size of the whole data
				if numpy.sum(proportions) > len(self.data):
					proportions[i] -= 1
				else:
					break

			for i in range(subset_count):
				sample = data.sample(n=proportions[i], replace=False)
				data = data.drop(sample.index)
				subsets[i] = subsets[i].append(sample)

		print("\n")
		return subsets

	def clean(self, discrete_columns=None):
		"""
		Removes data classifications which have to few instances to be used, based on a provided value (deletion_threshold).
		Also redistributes provided columns (discrete_columns) from one to many, such that each value in each column is
		converted into its own field, where it either has a value of 1 (if the original column-row combination contains
		the value of the field) or 0 (all other values).

		:param discrete_columns: list of column names to be broken into multiple fields (list of strings)
		"""
		cleaned_data = self.data

		if discrete_columns is not None:
			for column in discrete_columns:
				print("Distributing categorical data for column %s in the dataset" % column)

				for quality in cleaned_data[column].unique():  # Add every unique column value as its own feature.
					print("Creating new feature %s..." % quality)
					new_field = []
					print("\tFilling values in %s..." % quality)

					for row in cleaned_data[column]:  #1 where quality is present, 0 where it is not
						if row == quality:
							new_field += [1]
						else:
							new_field += [0]

					print("\tAdding %s to the dataset..." % quality)
					cleaned_data[quality] = new_field

				print("Dropping %s field from the dataset..." % column)
				cleaned_data.drop(column, axis=1, inplace=True)
				print("Feature distribution complete for the %s field!\n" % column)

		self.data = cleaned_data
		self.input_count = len(self.data.columns) - 1

	def normalize(self):
		"""
		Normalizes the data in the class instance.
		:rtype: object
		"""
		class_column = self.data[self.class_header]
		self.data = (self.data.drop(self.class_header, axis=1)).astype(float)
		self.data = (self.data - self.data.mean()) / (self.data.max() - self.data.min())
		self.data = pandas.concat((self.data, class_column), axis=1)

	def confusion(self, results):
		"""
		Generates confusion matricies.
		:rtype: object
		"""
		confusion_matrix = {'all': {'correct': 0, 'incorrect': 0}}
		for category in self.classes:
			confusion_matrix[category] = {}
			confusion_matrix[category]['true positive'] = 0
			confusion_matrix[category]['false positive'] = 0
			confusion_matrix[category]['true negative'] = 0
			confusion_matrix[category]['false negative'] = 0

		for classification in results:
			if classification[0] == classification[1]:
				confusion_matrix[classification[1]]['true positive'] += 1
				confusion_matrix['all']['correct'] += 1
				for category in confusion_matrix:
					if category != classification[1] and category != 'all':
						confusion_matrix[category]['true negative'] += 1
			elif classification[0] != classification[1]:
				confusion_matrix[classification[1]]['false negative'] += 1
				confusion_matrix[classification[0]]['false positive'] += 1
				confusion_matrix['all']['incorrect'] += 1
				for category in confusion_matrix:
					if category != classification[0] and category != classification[1] and category != 'all':
						confusion_matrix[category]['true negative'] += 1

		return confusion_matrix

	def analyze(self, testing_set, classifications, regression=False, verbose=False):
		"""
		Analyzes the data given classifications, by using the confusion matrix. Then calculates precision,
		recall, and F1. Can do classification or regression.
		:rtype: object
		"""
		print("\nANALYZING RESULTS") if verbose else None
		if not regression:
			confusion_matrix = self.confusion(classifications)
			correct = confusion_matrix['all']['correct']
			incorrect = confusion_matrix['all']['incorrect']
			accuracy = correct / (correct + incorrect)
			print("DATASET RESULTS:") if verbose else None
			print("CORRECT: %d\tINCORRECT: %d\tACCURACY: %f\n" % (correct, incorrect, accuracy))
			print("CATEGORICAL RESULTS:") if verbose else None
			for classification in confusion_matrix:
				if classification is not 'all':
					print(classification) if verbose else None
					tp = confusion_matrix[classification]['true positive']
					tn = confusion_matrix[classification]['true negative']
					fp = confusion_matrix[classification]['false positive']
					fn = confusion_matrix[classification]['false negative']

					print("\tTrue Positive: %s\n\tTrue Negative: %s\n\tFalse Positive: %s\n\tFalse Negative: %s\n"
						  % (tp, tn, fp, fn)) if verbose else None

					accuracy = tp / (tp + tn + fp + fn)

					if tp != 0 or fp != 0:
						precision = tp / (tp + fp)
					else:
						precision = 0

					if tp != 0 or fn != 0:
						recall = tp / (tp + fn)
					else:
						recall = 0

					if precision != 0 or recall != 0:
						f1 = 2 * ((precision * recall) / (precision + recall))
					else:
						f1 = 0

					if verbose:
						print("\tPrecision: %f\n\tRecall: %f\n\tF1: %f\n"
							  % (precision, recall, f1))
					else:
						print("Classification: %s\nAccuracy: %f\tPrecision: %f\tRecall: %f\tF1: %f\n"
							  % (classification, accuracy, precision, recall, f1))
		else:
			print("REGRESSION STATISTICS") if verbose else None
			n = 0
			squared_error_sum = 0
			actual_error_sum = 0
			for i in range(len(classifications)):
				n += 1
				error = float(testing_set.iloc[i][self.class_header]) - float(classifications.iloc[i])
				squared_error_sum += (error**2)
				actual_error_sum += error

			mean_actual_error = actual_error_sum / n
			mean_squared_error = squared_error_sum / n
			root_mean_squared_error = math.sqrt(mean_squared_error)

			squared_difference_sum = 0
			for i in range(len(classifications)):
				error = float(testing_set.iloc[i][self.class_header]) - float(classifications.iloc[i])
				squared_difference_sum += (error - mean_actual_error)**2

			std_deviation = math.sqrt(squared_difference_sum / n)

			if verbose:
				print("\tRoot Mean Squared Error (RMSE): %f\n\tMean Squared Error (MSE): %f\n\tMean Actual Error (MAE): %f\n\tStandard Deviation: %f\n"
					  % (root_mean_squared_error, mean_squared_error, mean_actual_error, std_deviation))
			else:
				print("Root Mean Squared Error (RMSE): %f\tMean Actual Error (MAE): %f"
					  % (root_mean_squared_error, mean_actual_error))

	def experimentFF(self, train, test, output_count, output_function, hidden_layers, learning_rate, regression, verbose, momentum):
		"""
		Initializes the feed forward network. and runs each required support method on said network. Sorts command line arguments.
		:rtype: object
		"""
		if hidden_layers == 0:
			print("TESTING FEED-FORWARD NETWORK WITH NO HIDDEN LAYERS") if verbose else None
			hidden_layer_array = None
		else:
			print("TESTING FEED-FORWARD NETWORK WITH %d HIDDEN LAYERS, EACH WITH %d NEURONS"
				  % (hidden_layers, self.input_count)) if verbose else None
			hidden_layer_array = [(self.input_count, 'sigmoid')] * hidden_layers

		net = FFNetwork(self.input_count,
						(output_count, output_function),
						self.data,
						hidden_layer_array,
						class_header=self.class_header,
						learning_rate=learning_rate,
						verbose=verbose,
						use_momentum=momentum,
						regression=regression)

		data_type = 'REGRESSION' if regression else 'CLASSIFICATION'
		print("TRAINING NETWORK WITH MOMENTUM FOR %s ON SIZE %d TRAINING SET" % (data_type, len(train))) if verbose else None
		train.apply(lambda row: net.train(row), axis=1)
		print("EVALUATING VALIDATION SET OF %d ENTRIES WITH TRAINED NETWORK" % len(test)) if verbose else None
		classifications = test.apply(lambda row:
									 net.classify(
										 row.drop(self.class_header),
										 row[self.class_header],
									 	 verbose=verbose),
									 axis=1)
		print("ANALYZING RESULTS") if verbose else None
		self.analyze(test, classifications, regression, verbose)

	def experimentRBF(self, train, test, cluster_count, reduction_method, learning_rate, regression, verbose, momentum):
		"""
		Handles running the RBF network.
		:param train: pandas DataFrame. training set.
		:param test: pandas DataFrame. testing set.
		:param cluster_count: int. number of clusters to be made.
		:param reduction_method: string. "condensed", "means", "medoids"
		:param learning_rate: float. learning rate multiplier.
		:param regression: Boolean. using regression or not.
		:param verbose: Boolean. lots of printed info vs some printed info. True is more.
		:param momentum: Boolean. using momentum or not.
		:return:
		"""
		net = RBFNetwork(train, cluster_count, self.data[self.class_header],
						 set_reduction_method=reduction_method,
						 regression=regression,
						 learning_rate=learning_rate,
						 verbose=verbose,
						 use_momentum=momentum)
		print("TESTING RADIAL BASIS FUNCTION NETWORK WITH %d GAUSSIAN FUNCTIONS" % len(net.function_layer)) if verbose else None
		cluster_count = len(net.prototypes)
		print("EVALUATING VALIDATION SET OF %d ENTREIS WITH TRAINED NETWORK" % len(test)) if verbose else None
		classifications = test.apply(lambda row: net.classify(row), axis=1)
		self.analyze(test, classifications, regression, verbose)
		return cluster_count

	def experiment(self, learning_rate=0.01, regression=False, verbose=False, momentum=False):
		"""
		Handles running the experiment for the feed forward network.
		:param learning_rate: float. learning rate multiplier.
		:param regression: Boolean. using regression or not.
		:param verbose: Boolean. lots of printed info vs some printed info. True is more.
		:param momentum: Boolean. using momentum or not.
		"""
		if regression:
			output_count = 1
			output_function = 'linear'
		else:
			output_function = 'sigmoid'
			output_count = len(self.classes)

		print('\nFEED FORWARD NETWORK\n')
		self.normalize()
		subsets = self.split(10, regression=regression)
		test = subsets[0]
		train = subsets[1]
		for subset in subsets[1:]:
			train = pandas.concat((train, subset), axis=0, ignore_index=True)

		print("TRAINING SET SIZE: %d\nVALIDATION SET SIZE: %d" % (len(train), len(test)))

		for i in range(3):
			self.experimentFF(train, test, output_count, output_function, i, learning_rate, regression, verbose, momentum)

		print('\nRADIAL BASIS FUNCTION NETWORK\n')
		if not regression:
			cluster_count = self.experimentRBF(train, test, 0, 'condensed', learning_rate, regression, verbose, momentum)
		else:
			cluster_count = math.ceil(len(train) / 4)

		self.experimentRBF(train, test, cluster_count, 'means', learning_rate, regression, verbose, momentum)
		self.experimentRBF(train, test, cluster_count, 'medoids', learning_rate, regression, verbose, momentum)
