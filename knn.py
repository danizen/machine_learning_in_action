import numpy as np
import operator


def create_dataset():
    group = np.array([[1.0,1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B' ]
    return group, labels


def classify0(inX, dataset, labels, k):
    dataset_size = dataset.shape[0]

    # calculate distance to each point
    diff_matrix = np.tile(inX, (dataset_size, 1)) - dataset
    diff_matrix = diff_matrix ** 2
    distances = diff_matrix.sum(axis=1) ** 0.5

    sorted_distance_idx = distances.argsort()

    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distance_idx[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    sorted_class_count = sorted(class_count.iteritems(), 
            key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]

