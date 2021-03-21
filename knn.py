import statistics
import pandas as pd
import numpy as np
from collections import Counter


def knn(k, testing_data, training_data):
    """
    :param k: k nearest neighbors
    :param testing_data: [[label_1, data_1], [label_2, data_2], ...]
    :param training_data: [[label_1, data_1], [label_2, data_2], ...]
    :return: classification accuracy
    """
    correct = 0
    total = len(testing_data)
    for i in testing_data:
        df = pd.DataFrame(columns=['label', 'dist'])
        for j in training_data:
            dist = np.linalg.norm(i[1] - j[1])
            df.loc[len(df), ] = [j[0], dist]
        df = df.sort_values(by='dist').reset_index(drop=True).loc[:k-1, ]
        try:
            if statistics.mode(df['label']) == i[0]:
                correct += 1
        except statistics.StatisticsError:
            correct += knn(k-1, [i], training_data)
    if total != 1:
        print(correct / total)
    return correct / total




