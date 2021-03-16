import numpy as np
import pandas as pd
from knn import knn


test = pd.read_csv('Bankruptcy/testing_zScore.csv', header=None)
train = pd.read_csv('Bankruptcy/training_zScore.csv', header=None)


def transform(data):
    result = []
    for i in range(len(data)):
        result.append((data.loc[i, ][64], np.array(data.loc[i, ][:64])))
    return result

testing = transform(test)
training = transform(train)


def find_center(data):
    """

    :param data: [(label_1, data_1), (label_2, data_2), ...]
    :return: center of each class in the form of
    """
    result = {}
    class_type = list(set([i[0] for i in data]))
    for i in class_type:
        total, count = [0] * 64, 0
        for d in data:
            if d[0] == i:
                count += 1
                total = [sum(i) for i in list(zip(total, d[1]))]
        result[i] = [i / count for i in total]
    print(result)
    return result



def find_within_var(data):
    result = 0
    center = find_center(data)
    for d in data:
        arr = np.array(d[1]) - np.array(center[d[0]])
        result += np.outer(arr, arr)
    return result


def find_between_var(data):
    class_type = list(set([i[0] for i in data]))
    sample_center = [0] * 64
    for d in data:
        sample_center = [sum(i) for i in list(zip(sample_center, d[1]))]
    sample_center = [i / len(data) for i in sample_center]
    class_center = find_center(data)
    result = 0
    for j in class_type:
        count = sum([i[0] == j for i in data])
        arr = np.array(class_center[j]) - np.array(sample_center)
        result += count * np.outer(arr, arr)
    return result

