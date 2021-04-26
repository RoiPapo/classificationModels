import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd
import collections as cl


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p
        self.points = {}
        self.points_label = []

        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (316327451, 206230021)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        T

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.his method trains a k-NN classifier on a given training set X with label set y.
        """

        clustered_labels = {label: [] for label in y}
        for i in range(X.shape[0]):
            clustered_labels[y[i]].append(X[i, :])
            self.points_label.append((X[i, :], y[i]))
        self.points = clustered_labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        predicteded_labels = []
        current_point_distances = []

        for test_point in X:
            closest_point_by_class = {label: [] for label in self.points.keys()}
            current_point_distances = []
            for train_point in self.points_label:
                dist = minkowskiDist(test_point, train_point[0], self.p)
                current_point_distances.append((dist, train_point[1]))
            current_point_distances.sort(key=lambda tup: tup[0])
            k_neighbors = current_point_distances[:self.k]
            closest_point_by_class[k_neighbors[0][1]] = [k_neighbors[0]]  # nearest neighbor
            curr_labels = [k_neighbors[0][1]]
            for neighbor in k_neighbors:
                if neighbor[1] not in curr_labels:
                    curr_labels.append(neighbor[1])
                    closest_point_by_class[neighbor[1]].append(neighbor)
            k_labels = [x[1] for x in k_neighbors]
            predicted_label = most_frequent(k_labels, closest_point_by_class)
            predicteded_labels.append(predicted_label)
        return np.array(predicteded_labels)
        ### Example code - don't use this:
        # return np.random.randint(low=0, high=2, size=len(X), dtype=np.uint8)


def minkowskiDist(v1, v2, p):
    return (np.abs(v1 - v2) ** p).sum(axis=0) ** (1 / p)


def most_frequent(List, closest_point_by_class):
    """
        This method computes the label based on most frequent neighbors
        and break the tie if there is one

        :param List: array of labels of k close neighbors
        :return: closest_point_by_class dict of colsest neighbor by class.
        """
    occurences_list = cl.Counter(List)
    number_of_ties = 1
    for i in range(len(occurences_list) - 1):
        if occurences_list[i] == occurences_list[i + 1]:
            number_of_ties = +1
    if number_of_ties > 1:
        # ------------- tie braker------------
        maximum_neighbors = [x[0] for x in occurences_list.most_common(number_of_ties)]
        tie_braker_winner = (100000, 100000)  # a big number , to initiate
        for group in maximum_neighbors:
            if closest_point_by_class[group][0][0] < tie_braker_winner[0]:
                tie_braker_winner = closest_point_by_class[group][0]
            elif closest_point_by_class[group][0][0] == tie_braker_winner[0]:
                alfabetic = min(tie_braker_winner[1], closest_point_by_class[group][0][1])
                tie_braker_winner = closest_point_by_class[alfabetic][0]
        return tie_braker_winner[1]
    # ------------- tie braker------------
    else:
        return occurences_list.most_common(1)[0][0]


def main():
    print("*" * 20)
    print("Started HW1_206230021_316327451.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
