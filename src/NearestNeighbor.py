import pandas
import numpy
from collections import Counter
import multiprocessing
import itertools
import math


class NearestNeighbor:

    @staticmethod
    def kNearestNeighbor(testing_set, training_set, k=3, class_header="Class"):
        """
        Base K Nearest Neighbor algorithm.

        :param testing_set: pandas DataFrame. Set to classify based on the training set
        :param training_set: pandas DataFrame. Set used to classify the testing set
        :param k: int. number of nearest neighbors to test against
        :param class_header: string. Name of the column which contains the classifications for each dataset.
        :return: List. Class guesses for every point in the testing set
        """
        pool = multiprocessing.Pool()
        multiprocess_count = multiprocessing.cpu_count()
        set_list = []
        partition_size = math.ceil(len(testing_set) / multiprocess_count)
        for i in range(multiprocess_count - 1):  # repeat for every subset
            sample = testing_set.iloc[i * partition_size: (i + 1) * partition_size]
            set_list.append((sample, training_set, k, class_header))
        set_list.append((testing_set.iloc[(multiprocess_count - 1) * partition_size:], training_set, k, class_header))

        # print(set_list)

        distances = list(
            itertools.chain.from_iterable(pool.map(NearestNeighbor.kNearestNeighborMultiprocess, set_list)))
        pool.close()
        pool.join()
        # print(distances)
        return distances

    @staticmethod
    def kNearestNeighborMultiprocess(set_list):
        testing_set = set_list[0]
        training_set = set_list[1]
        k = set_list[2]
        class_header = set_list[3]
        classifications = []  # list of each guess for each element in the test set
        distances = testing_set.apply(
            lambda testing_row: training_set.apply(
                lambda training_row: NearestNeighbor.calcDistance(testing_row, training_row, class_header),
                axis=1),
            axis=1)
        distances = (distances.to_numpy())
        k_distances = []
        for distance in distances:
            k_distances.append(sorted(distance, key=lambda t: t[0])[:k])
        for entry in k_distances:
            classifications.append(Counter([x[1] for x in entry]).most_common(1)[0][0])

        return classifications

    @staticmethod
    def setMerge(dataframes):
        """
        Support method which takes in a list of dataframes and merges all but one into the training set.
        The last becomes the testing_set

        :param dataframes: List of pandas DataFrame.
        :return: pandas DataFrame. merged_training_set. all but one dataframe merged into one.
                 pandas DataFrame. testing_set. The first dataframe in the list.
        """
        merged_training_set = pandas.DataFrame(columns=dataframes[0].columns)
        testing_set = pandas.DataFrame(
            columns=dataframes[0].columns)  # Set column names of empty test and train dataframes
        counter = 0
        for dataframe in dataframes:
            if counter == 0:
                testing_set = dataframe  # If last dataframe add to test set.
            else:  # Otherwise add to train set
                merged_training_set = merged_training_set.append(dataframe, ignore_index=True)
            counter += 1

        return merged_training_set, testing_set

    @staticmethod
    def oneNearestNeighbor(testing_point, classifier_set, class_header="Class", return_distance=False):
        """
        Support method for condensed nearest neighbor to compare one series to a set

        :param testing_point: Series. Point to be classified based on classifier set
        :param classifier_set: pandas DataFrame. set by which to classify the testing point
        :param class_header: string. Name of the column which contains the classifications for each dataset.
        :return: String. Predicted testing_point classification
        """
        # Fast nearest node finder, lambda sets each row to distance from selected point
        distances = classifier_set.apply(lambda row: NearestNeighbor.calcDistance(testing_point, row, class_header),
                                         axis=1)
        nearest = sorted(distances, key=lambda t: t[0])[0]  # Select lowest distance
        classification = nearest[1]  # Set class to class of nearest point

        if return_distance:
            return nearest[0]
        return classification

    @staticmethod
    def calcDistance(row_a, row_b, class_header="Class"):
        """
        Method that finds the Euclidean distance between two points (rows)

        :param row_a: pandas DataFrame. One row which has integer or float values for each of its cells
        :param row_b: pandas DataFrame. One row which has integer or float values for each of its cells,
                      must have same number of columns and the same column names as row_a
        :param class_header: string. Name of the column which contains the classifications for each dataset.
        :return: Float. The Euclidean distance between the two rows
        """
        classification = row_b[class_header]  # set return class to class of row b
        row_a = row_a.drop(labels=class_header)  # Drop classes from each row
        row_b = row_b.drop(labels=class_header)
        distance = numpy.linalg.norm(row_a.values - row_b.values)  # Fast distance using numpy

        return [distance, classification]

    @staticmethod
    def deleteOutliers(training_set, k=3, class_header="Class"):
        """
        Support method that deletes outliers from the given set

        :param k: int. how many nearby points to compare with
        :return: pandas DataFrame. new set without outliers
        :param class_header: string. Name of the column which contains the classifications for each dataset.
        :param training_set: pandas DataFrame
        """

        nodes_removed = 0
        nodes_checked = 0

        print("Checking for outlier")

        for index, row in training_set.iterrows():

            row_set = pandas.DataFrame(columns=training_set.columns)
            row_set = row_set.append(row)

            tmp_training_set = training_set.drop(index)  # Creates a new training set without the current point

            if NearestNeighbor.kNearestNeighbor(row_set, tmp_training_set, k, class_header)[0] != row[
                class_header]:  # if the point is not classified correctly
                training_set = tmp_training_set  # the point is removed from the training set as it is an outlier
                nodes_removed += 1
                return training_set
            else:
                nodes_checked += 1

        print("Number of points removed: %d" % nodes_removed)
        print("Number of points checked: %d" % nodes_checked)
        return training_set

    @staticmethod
    def outputAnalyzer(testing_set, predictions, class_header):
        """
        Calculates confusion matrix for a specific class
        :param testing_set: pandas DataFrame. Test set with known classes
        :param predictions: String[]. Array of predicted classes
        :param class_header: String. Column name of classes
        :return: {string:{string:int}}. Dictionary representing number of predictions for each pair of classes
        """
        actuals = list(testing_set[class_header])  # List out true classes
        confusion_matrix = {}
        for primary in set(actuals):  # For each class
            confusion_matrix[primary] = {}
            for secondary in set(actuals):  # For each class pair
                confusion_matrix[primary][secondary] = 0  # initialize nested dictionary item

        for i in range(len(predictions)):  # For each prediction
            confusion_matrix[actuals[i]][predictions[i]] += 1  # increment appropriate value

        return confusion_matrix

    @staticmethod
    def condensedNearestNeighbor(training_set, class_header="Class"):  # page 22 of L3-NonParametric
        """
        Support method that takes in a training set and produces a condensed training set.

        :param training_set: pandas DataFrame. Training set to be condensed.
        :param class_header: string. Name of the column which contains the classifications for each dataset.
        :return: pandas DataFrame. Condensed training set.
        """

        training_set = training_set.sample(frac=1).reset_index(
            drop=True)  # mixes the order of the rows for random order iteration

        has_z_changed = True
        absorbed_points = 0

        condensed_training_set = pandas.DataFrame(
            columns=training_set.columns)  # create a new empty training set for prototypes (Z)

        print("Condensing Dataset")

        while has_z_changed:
            has_z_changed = False

            for index, row_tuple in enumerate(training_set.iterrows()):  # for every point
                row = row_tuple[1]
                if index == 0:
                    condensed_training_set = condensed_training_set.append(row)

                if NearestNeighbor.oneNearestNeighbor(row, condensed_training_set, class_header) != row[
                    class_header]:  # if the point is incorrectly classified
                    condensed_training_set = condensed_training_set.append(row)  # add the point to the CTS
                    has_z_changed = True
                else:
                    absorbed_points += 1

        print("Number of points absorbed: %d" % absorbed_points)
        print("Number of points in training set: %d" % len(training_set.index))
        print("Condensed Training Set Size: %d" % len(condensed_training_set.index))

        return condensed_training_set  # returns the condensed training set

    @staticmethod
    def getPerformance(prediction_list, testing_set, class_header="Class"):
        """
        Support method which takes in a list of predicted class and a
        testing set and returns a float representing the performance.

        :param prediction_list: List of Strings which represent the predicted class values, in order, of the testing set
        :param testing_set: pandas DataFrame. includes all correct class values to be compared to the predicted values.
        :param class_header: string. Name of the column which contains the classifications for each dataset.
        :return: Float. number of classes correctly classified divided by the total number of classes.
        """
        index = 0
        correct = 0
        total = 0
        for row in testing_set.iterrows():  # For each row in test set
            if row[1][class_header] == prediction_list[index]:  # if prediction is correct
                correct += 1
                total += 1
            else:
                total += 1
            index += 1
        performance = correct / total  # Calculate ratio of correct predictions
        return performance

    @staticmethod
    def editedNearestNeighbor(testing_set, training_set, k=3,
                              class_header="Class"):  # page 21 of L3-NonParametric (option 1)
        """
        Edited Nearest Neighbor method which tries to optimize the success rate of knn by dropping outlier points

        :param testing_set:
        :param k: k used for testing kNN while running editedNearestNeighbor, should mirror k from baseline test
        :param training_set: pandas DataFrame. set that will be modified to get the best success rate of knn
        :param class_header: string. Name of the column which contains the classifications for each dataset.
        :return: pandas DataFrame. Newly edited training set where performance increased compared to base kNN
        """

        training_set = training_set.sample(frac=1).reset_index(
            drop=True)  # mixes the order of the rows for random order iteration

        edited_training_set = training_set.copy()
        lost_performance = False
        base_predict_list = NearestNeighbor.kNearestNeighbor(testing_set, training_set, k, class_header)
        baseline_performance = NearestNeighbor.getPerformance(base_predict_list, testing_set, class_header)
        print("k Nearest Neighbor Performance: ", baseline_performance)

        while not lost_performance:

            # Find an outlier to delete, k is all other points left in the edited training set
            tmp_edited_training_set = NearestNeighbor.deleteOutliers(edited_training_set,
                                                                     k=(len(edited_training_set.index) - 1),
                                                                     class_header=class_header)
            # Get the class prediction list based on this new dataset
            tmp_predicted_list = NearestNeighbor.kNearestNeighbor(testing_set, tmp_edited_training_set, k, class_header)
            # Get the performance based on the new dataset
            curr_performance = NearestNeighbor.getPerformance(tmp_predicted_list, testing_set, class_header)

            # If the entire dataset is gone through without a deletion, there are no more changes to be made, break
            if len(edited_training_set) == len(tmp_edited_training_set):
                break

            print("Current Performance: ", curr_performance)

            # Test the base performance against the most recent performance
            if baseline_performance > curr_performance:
                lost_performance = True  # If the old performance is better
                print("Stopping: Performance Lost")
            else:  # If the new performance is better
                edited_training_set = tmp_edited_training_set
                baseline_performance = curr_performance

        print("\nEdited Training Set Size: ", len(edited_training_set.index))
        return edited_training_set
