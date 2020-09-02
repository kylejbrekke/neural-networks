import pandas
import numpy
import math
import multiprocessing
from itertools import zip_longest

MAX_ITERATIONS = 100
SETTLE_THRESHOLD = 1


class Cluster:
    @staticmethod
    def byMedoids(dataset, number_of_clusters, class_header="Class", verbosity=0, return_clusters=False):
        """
        Find Medoids using Partitioning Around Medoids to cluster data.
        :param dataset: pandas DataFrame. The Data to cluster
        :param number_of_clusters: int. The number of clusters
        :param class_header: String. The name of the class column in the dataset
        :param verbosity: int. How much debug information to print to console, [0-3]
        :return: pandas DataFrame. Selected Medoids
        """
        medoids = dataset.sample(number_of_clusters)  # randomly select medoids from dataset

        if verbosity >= 1:
            print("INITIAL MEDOIDS")
            print(medoids)
            if verbosity >= 2:
                print("DATAFRAME DATASET")
                print(dataset)

        for iterations in range(MAX_ITERATIONS):  # Loop until MAX_ITERATIONS or settled
            if verbosity >= 1:
                print("ITERATIONS")
                print(iterations)

            clusters = Cluster.calcClusters(dataset, medoids, number_of_clusters, verbosity=verbosity,
                                            class_header=class_header)  # Assign all points to a cluster

            base_distortion = Cluster.calcDistortion(medoids, clusters, class_header=class_header)
            # Find base distortion

            set_list = []  # set up multiprocessing structures
            work_list = []
            change_list = []

            for medoid_row_index, medoid_tuple in enumerate(medoids.iterrows()):  # For each medoid
                medoid_frame_index = medoid_tuple[0]
                for datum_index, datum in clusters[medoid_row_index].iterrows():  # For each point in the medoid cluster
                    if medoid_frame_index != datum_index:  # Do not try to swap a medoid with itself
                        temp = medoids.copy()  # Make a copy of the medoids DataFrame
                        temp.iloc[medoid_row_index] = datum  # Swap the medoid in the copy
                        temp.index.values[medoid_row_index] = datum.name
                        work_list.append((temp, clusters, class_header))  # add calculation arguments to work list
                        change_list.append((medoid_row_index, datum))  # add swap info to change list

            multiprocess_count = multiprocessing.cpu_count()  # Find cpu count
            partition_size = math.ceil(len(work_list) / multiprocess_count)  # find size of work list partitions
            if verbosity >= 1:  # optionally print work list length
                print("Work list length:")
                print(len(work_list))
            for i in range(multiprocess_count - 1):  # repeat for every subset
                sample = work_list[i * partition_size: (i + 1) * partition_size]  # take a subset of the work list
                set_list.append(sample)  # add that subset as an item in the set list
            set_list.append((work_list[(multiprocess_count - 1) * partition_size:]))  # add tailing subset to set list
            if verbosity > 2:  # optionally print entire set list.
                print("Set list")
                print(set_list)
            pool = multiprocessing.Pool(processes=multiprocess_count)  # create multiprocessing pool
            distortion_lists = pool.map(Cluster.calcDistortionList, set_list)  # map set list to processing pool
            pool.close()
            pool.join()
            #print(distortion_lists)
            distortions = sum(distortion_lists, [])
            #print(distortions)

            break_flag = True  # set break flag in case there are no good changes
            distortion_index = 0
            for medoid_row_index, _ in enumerate(medoids.iterrows()):  # For each medoid
                cluster_size = len(clusters[medoid_row_index])
                distortions_subset = distortions[distortion_index: distortion_index + cluster_size]
                distortion_index += cluster_size  # keep track of how far we are through the change list
                if len(distortions_subset) != 0:  # did this cluster have any possible changes
                    best_distortion = min(distortions_subset)  # pick the best distortion
                    if best_distortion < base_distortion:  # if that distortion is better than our old distortion
                        best_dist_index = distortions.index(best_distortion)
                        best_change = change_list[best_dist_index]  # apply the change for that distortion.
                    else:
                        best_change = None
                else:
                    best_change = None
                if verbosity > 0:  # Optionally print best changes
                    print("MEDOIDS")
                    print(medoids)
                    print("BEST_CHANGE")
                    print(best_change)
                if best_change is not None:  # make sure there is a change before trying to make it.
                    medoids.iloc[best_change[0]] = best_change[1]  # swap best change into medoids list
                    medoids.index.values[best_change[0]] = best_change[1].name
                    break_flag = False

            if break_flag:  # if we made no changes then the clustering is settled.
                break

        medoids = medoids.drop_duplicates()  # make sure we do not duplicate medoids
        if return_clusters is True:  # optionally return clusters
            return medoids, clusters
            pass
        else:
            return medoids  # return medoids dataframe

    @staticmethod
    def calcDistortionList(work_list):
        """
        Perform distortion calculations on a work list
        :param work_list: list of tuples, the list calculations to do
        :return: list of floats, the results of each calculation
        """
        distortion_list = []
        for swap in work_list:
            distortion_list.append(Cluster.calcDistortion(*swap))  # call calcDistortion with tuple expansion as args
        return distortion_list


    @staticmethod
    def calcDistortion(medoids, clusters, class_header="Class"):
        """
        Find the distortion of a set of medoids
        :param medoids: pandas DataFrame. The medoids
        :param clusters: pandas DataFrame[]. List of clusters centered on each medoid
        :param class_header: String. The name of the class column in the dataset
        :return: float. The distortion value of this configuration
        """
        distortion = 0
        for medoid_row_index, medoid_tuple in enumerate(medoids.iterrows()):  # For every Medoid
            for _, datum in clusters[medoid_row_index].iterrows():  # For each point in the medoid cluster
                # Add the distance between medoid and data point squared to total distortion
                distortion += (Cluster.calcDistance(medoid_tuple[1], datum, class_header=class_header)) ** 2
        return distortion

    @staticmethod
    def calcAvgDistances(centroids, clusters, class_header="Class"):
        """
        Find the average distances of items in clusters to their centroids.
        :param centroids: pandas DataFrame of cluster centroids
        :param clusters:  pandas DataFrame[], list of cluster members
        :param class_header: String. The name of the class column in the dataset
        :return: float[], list of averages, indexed similarly to centroids and clusters
        """
        avg_distances = [0] * len(centroids)
        multiprocess_count = multiprocessing.cpu_count()  # Find processor count
        for centroid_row_index, centroid_tuple in enumerate(centroids.iterrows()):  # For each cluster
            work_list = []  # initialize multiprocessing structures
            set_list = []
            for _, datum in clusters[centroid_row_index].iterrows():  # For each point in the medoid cluster
                work_list.append((centroid_tuple[1], datum, class_header))  # add calculation to work list

            partition_size = math.ceil(len(work_list) / multiprocess_count)  # find size of each work subeset
            for i in range(multiprocess_count - 1):  # repeat for every subset
                sample = work_list[i * partition_size: (i + 1) * partition_size]  # break work list into fair subsets
                set_list.append(sample)
            set_list.append((work_list[(multiprocess_count - 1) * partition_size:]))
            pool = multiprocessing.Pool(processes=multiprocess_count)  # create multiprocessing pool
            # calculate sum of list of all distances from work list tasks
            avg_distances[centroid_row_index] = sum(sum(pool.map(Cluster.calcDistanceList, set_list), []))
            pool.close()
            pool.join()

            if avg_distances[centroid_row_index] is not 0:  # make sure we do not divide by 0
                # calculate average of distance list
                avg_distances[centroid_row_index] = avg_distances[centroid_row_index] / len(clusters[centroid_row_index])
        return avg_distances

    @staticmethod
    def calcClusters(dataset, medoids, number_of_clusters, verbosity=0, class_header="Class"):
        """
        Calculate clusters centered on list of points (medoids)
        :param dataset: pandas DataFrame. The dataset to be clustered
        :param medoids: pandas DataFrame. The points to cluster around
        :param number_of_clusters: int. The number of clusters to make
        :param verbosity: int. How much debug information to print to console, [0-3]
        :param class_header: String. The name of the class column in the dataset
        :return: pandas DataFrame[]. List of clusters centered on each point (medoid)
        """
        clusters = [pandas.DataFrame(columns=dataset.columns)] * number_of_clusters  # create array of clusters
        multiprocess_count = multiprocessing.cpu_count()  # Find processor count
        pool = multiprocessing.Pool(processes=multiprocess_count)  # create multiprocessing pool

        set_list = []
        partition_size = math.ceil(len(dataset) / multiprocess_count)
        for i in range(multiprocess_count - 1):  # repeat for every subset
            sample = dataset.iloc[i * partition_size: (i + 1) * partition_size]  # take a sample of data
            set_list.append((sample, medoids, number_of_clusters, verbosity, class_header))  # fill work list
        set_list.append(
            (dataset.iloc[(multiprocess_count - 1) * partition_size:], medoids, number_of_clusters, verbosity, class_header))

        # find list of clustering for each subset
        clusters_subsets = pool.starmap(Cluster.calcClustersMultiprocess, set_list)
        pool.close()
        pool.join()
        # Transpose 2d list of dataframes so each lower level list is of the same cluster
        cluster_lists = [[i for i in element if i is not None] for element in list(zip_longest(*clusters_subsets))]

        for i in range(number_of_clusters):  # concat together each list of cluster subsets.
            clusters[i] = pandas.concat(cluster_lists[i])
        return clusters

    @staticmethod
    def calcClustersMultiprocess(dataset, medoids, number_of_clusters, verbosity=0, class_header="Class"):
        """
        Calculate clusters centered on list of points (medoids)
        :param dataset: pandas DataFrame. The dataset to be clustered
        :param medoids: pandas DataFrame. The points to cluster around
        :param number_of_clusters: int. The number of clusters to make
        :param verbosity: int. How much debug information to print to console, [0-3]
        :param class_header: String. The name of the class column in the dataset
        :return: pandas DataFrame[]. List of clusters centered on each point (medoid)
        """
        clusters = [pandas.DataFrame(columns=dataset.columns)] * number_of_clusters  # create array of clusters
        for _, datum in dataset.iterrows():  # For every datum
            nearest_medoid_index = 0
            nearest_medoid = next(medoids.iterrows())[1]
            shortest_distance = Cluster.calcDistance(datum, nearest_medoid,
                                                     class_header=class_header)  # Find nearest medoid
            for medoid_row_index, medoid_tuple in enumerate(medoids.iterrows()):
                medoid_frame_index = medoid_tuple[0]  # Find nearest medoid
                medoid = medoid_tuple[1]
                if medoid_row_index is 0: continue
                distance = Cluster.calcDistance(datum, medoid,
                                                class_header=class_header)  # find distance to current medoid
                if verbosity >= 2:
                    print("DISTANCE TO", medoid_frame_index)
                    print(distance)
                    print("MEDOID INDEX")
                    print(medoid_row_index)

                if distance < shortest_distance:  # if current medoid is closer than all previous select it
                    shortest_distance = distance
                    nearest_medoid_index = medoid_row_index

            if verbosity >= 3:
                print("ITERROW DATUM")
                print(datum)
                print("DATAFRAME ARRAY CLUSTERS")
                print(clusters)

            # Assign datum to appropriate cluster
            clusters[nearest_medoid_index] = clusters[nearest_medoid_index].append(datum)
        return clusters

    @staticmethod
    def byMeans(dataset, number_of_clusters, class_header="Class", verbosity=0, return_clusters=False):
        """
        Find Means using means clustering
        :param dataset: pandas DataFrame. The Data to cluster
        :param number_of_clusters: int. The number of clusters
        :param class_header: String. The name of the class column in the dataset
        :param verbosity: int. How much debug information to print to console, [0-3]
        :return: pandas DataFrame. Selected Means
        """
        if verbosity >= 2:  # optionally print dataset shape and info
            print(dataset.shape)
            print(dataset)

        old_dataset = dataset.copy()
        dataset = dataset.drop(columns=class_header)  # remove non-float class column

        # Assign centroids to random values which fit into dataset space.
        centroids = pandas.DataFrame(columns=dataset.columns,
                                     data=numpy.random.uniform(dataset.min(), dataset.max(),
                                                               (number_of_clusters, dataset.shape[1])))
        if verbosity >= 1:  # optionally print centroids and random dataset
            print("INITIAL CENTROIDS")
            print(centroids)
            if verbosity >= 2:
                print("DATAFRAME DATASET")
                print(dataset)

        for iterations in range(MAX_ITERATIONS):  # Loop until MAX_ITERATIONS or settled
            if verbosity >= 1:  # optionally print iteration count
                print("ITERATIONS")
                print(iterations)

            # calculate clustering of data
            clusters = Cluster.calcClusters(dataset, centroids, number_of_clusters, verbosity=verbosity)

            old_centroids = centroids.copy()  # copy centroid dataframe

            if verbosity >= 2:  # optionally print cluster list
                print("DATAFRAME ARRAY CLUSTERS")
                print(clusters)

            for cluster_index, cluster in enumerate(clusters):  # Calculate new centroids
                cluster_mean = cluster.mean()
                if not cluster_mean.isnull().any():  # make sure we dont write null means to centroid list
                    centroids.loc[cluster_index] = cluster_mean

            if verbosity >= 1:
                print("OLD CENTROIDS")
                print(old_centroids)
                print("NEW CENTROIDS")
                print(centroids)

            if old_centroids is not None:  # Calculate sum of centroid movements.
                centroid_change = 0
                for centroid_index, centroid in centroids.iterrows():
                    centroid_change += abs(Cluster.calcDistance(centroid, old_centroids.loc[centroid_index]))

                if verbosity >= 1:
                    print("CENTROID DIFF")
                    print(centroid_change)

                if centroid_change < SETTLE_THRESHOLD:  # break if centroid movement is below threshold.
                    break

        # Final Cluster re-calculation
        clusters = Cluster.calcClusters(old_dataset, centroids, number_of_clusters,
                                        verbosity=verbosity, class_header=class_header)
        # Create new dataframe with class column of and row for each centroid
        centroids_class = pandas.DataFrame(data=["NOCLASS"] * centroids.shape[0], columns=[class_header])
        if verbosity >= 2:
            print(centroids_class)
            print(centroids)
        for cluster_index, cluster in enumerate(clusters):  # For each cluster
            if verbosity >= 2:
                print(cluster_index)
                print(cluster)
            if cluster.size > 0:  # If cluster is not empty set centroid class to most common class in cluster
                centroids_class.iat[cluster_index, 0] = cluster.mode().loc[0][0]
        if old_dataset.columns[0] == class_header:  # check if class column should be first or last.
            print("CLASS IS FIRST COL")
            centroids = pandas.concat([centroids_class, centroids], axis=1)  # merge class to centroids as first column
        else:
            print("CLASS IS NOT FIRST COL")
            centroids = pandas.concat([centroids, centroids_class], axis=1)  # merge class to centroids as last column
        for centroid in centroids.iterrows():  # For each centroid
            if centroid[1][class_header] is "NOCLASS":  # Trim NOCLASS centroids (empty cluster)
                centroids = centroids.drop(centroid[0])
        centroids = centroids.reset_index(drop=True)  # Reindex centroids

        if return_clusters is True:  # optionally return cluster list
            return centroids, clusters
            pass
        else:
            return centroids  # return centroids dataframe

    @staticmethod
    def calcDistanceList(work_list):
        """
        Calculate distances for a work list
        :param work_list: List of distances to calculate
        :return: float[], list of distances
        """
        distance_list = []
        for swap in work_list:  # for every work item find distance
            distance_list.append(Cluster.calcDistance(*swap))
        return distance_list

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
        if class_header in row_a:  # check for header on each row and drop it.
            row_a = row_a.drop(labels=class_header)
        if class_header in row_b:
            row_b = row_b.drop(labels=class_header)
        distance = numpy.linalg.norm(row_a.values - row_b.values)  # find euclidean distance as vectors

        return distance


if __name__ == "__main__":
    df = pandas.DataFrame(numpy.random.randint(0, 10, size=(100, 7)), columns=list('ABCDEFZ'))
    #df = pandas.DataFrame(columns=("x", "y", "Z"),
    #                      data={(1, 1, "a"), (1, 2, "a"), (2, 1, "a"), (6, 6, "b"), (6, 7, "b"), (7, 6, "b"),
    #                            (10, 10, "c"), (10, 11, "c"), (11, 10, "c")})
    print(df)
    c, cl = Cluster.byMedoids(df, 3, class_header="Z", verbosity=0, return_clusters=True)

    print(Cluster.calcAvgDistances(c, cl, class_header="Z"))
    print(type(c))
    print(c)
