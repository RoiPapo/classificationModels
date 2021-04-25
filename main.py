import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def generateSamples(num):
    E1 = np.array([-1, 1])
    E2 = np.array([-2.5, 2])
    cov = np.identity(2)
    list_of_samples = []
    class_color = np.zeros(num)
    for i in range(num):
        rand = np.random.random()
        if rand >= 0.5:
            sample = np.random.multivariate_normal(E1, cov)
            class_color[i] = 1
        else:
            sample = np.random.multivariate_normal(E2, cov)
        list_of_samples.append(sample)
    return list_of_samples, class_color


def displayErrorGraph(line_1_Y, line_2_Y, labelX, labelY):
    # plt.plot(range(1, 21), line_1_Y, label="Train Error")
    # plt.plot(range(1, 21), line_2_Y, label="Test Error")
    plt.plot([10, 15, 20, 25, 30, 35, 40], line_1_Y, label="Train Error")
    plt.plot([10, 15, 20, 25, 30, 35, 40], line_2_Y, label="Test Error")
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.legend()
    plt.show()


def createScatterPlot(list_of_samples, labels):
    sample_class1 = []
    sample_class2 = []
    for i in range(len(labels)):
        if labels[i] == 1:
            sample_class1.append(list_of_samples[i])
        else:
            sample_class2.append(list_of_samples[i])
    s1 = np.array(sample_class1)
    s2 = np.array(sample_class2)
    plt.scatter(s1[:, 0], s1[:, 1], c='grey', label='First Distribution')
    plt.scatter(s2[:, 0], s2[:, 1], c='green', label='Second Distribution')
    plt.xlabel("1st entry")
    plt.ylabel("2nd entry")
    plt.legend()
    plt.show()


def calculate_error(prediction, labels):
    cer = 0
    length = len(labels)
    for i in range(length):
        if prediction[i] != labels[i]:
            cer += 1
    return cer / length


def main():
    list_of_samples, train_labels = generateSamples(400)
    # createScatterPlot(list_of_samples, train_labels)
    list_of_TEST, TEST_true_labels = generateSamples(100)
    # createScatterPlot(list_of_TEST, TEST_true_labels)

    k = 1
    x_cols = ['sepal-length', 'sepal-width']
    y_cols = "label_num"
    Train_Errors = []
    Test_Errors = []
    # for k in range(1, 21):
    #     # Instantiate the classifier
    #     model = KNeighborsClassifier(n_neighbors=k)
    #     # Fit on the training set
    #     model.fit(list_of_samples, train_labels)
    #     prediction = model.predict(list_of_TEST)
    #     test_error = calculate_error(prediction, TEST_true_labels)
    #     Test_Errors.append(test_error)
    #     # print("test Error is: ", test_error)
    #     prediction = model.predict(list_of_samples)
    #     train_error = calculate_error(prediction, train_labels)
    #     Train_Errors.append(train_error)
    for train_size in [10, 15, 20, 25, 30, 35, 40]:
        list_of_samples, train_labels = generateSamples(train_size)
        list_of_TEST, TEST_true_labels = generateSamples(100)
        # Instantiate the classifier
        model = KNeighborsClassifier(n_neighbors=10)
        # Fit on the training set
        model.fit(list_of_samples, train_labels)
        prediction = model.predict(list_of_TEST)
        test_error = calculate_error(prediction, TEST_true_labels)
        Test_Errors.append(test_error)
        # print("test Error is: ", test_error)
        prediction = model.predict(list_of_samples)
        train_error = calculate_error(prediction, train_labels)
        Train_Errors.append(train_error)

    displayErrorGraph(Train_Errors, Test_Errors, "TrainSize", "Error")


if __name__ == "__main__":
    main()
