import random
import numpy as np
from knn import knn
from imgRead import readImages


path = 'new-images'
labels, subjects = readImages(path)


def random_select(n):
    test, train = [], []
    for i in range(1, 16):
        key = 'subject' + f"{i:02}"
        vec = random.sample(range(0, 11), n)
        for j in range(11):
            if j in vec:
                train.append([labels[key][j], np.array(subjects[key][j])])
            else:
                test.append([labels[key][j], np.array(subjects[key][j])])
    return test, train

testing, training = random_select(8)

knn(1, testing, training)
