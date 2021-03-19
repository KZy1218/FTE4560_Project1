from knn import *
from linear_discriminant_analysis import *



def main():
    testing = transform(pd.read_csv('Bankruptcy/testing_zScore.csv', header=None))
    training = transform(pd.read_csv('Bankruptcy/training_zScore.csv', header=None))
    between = find_between_var(testing)
    within = find_within_var(testing)
    w = np.real(projector(between, within))

    for data in testing:
        data[1] = np.dot(np.array(data[1]), w)
    for data in training:
        data[1] = np.dot(np.array(data[1]), w)

    knn(1, testing, training)
    knn(3, testing, training)
    knn(5, testing, training)
    knn(7, testing, training)

# transform original data using the projector and apply knn again

if __name__ == '__main__':
    main()

