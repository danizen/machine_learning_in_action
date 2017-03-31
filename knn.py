import numpy as np
import operator
import pandas as pd
import glob
from os.path import basename


def create_dataset():
    group = np.array([[1.0,1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B' ]
    return group, labels


def load_dataset(filename):
    df = pd.read_csv(filename, sep='\s+', header=None)
    labels = df[3].as_matrix()
    matrix = df.drop(3, axis=1).as_matrix()
    return matrix, labels


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


def autonorm(matrix):
    minvals = matrix.min(0)
    ranges = matrix.max(0) - minvals
    norm_matrix = (matrix - minvals) / ranges
    return norm_matrix, ranges, minvals


def datingClassTest(filename):
    hoRatio = 0.10

    matrix, labels = load_dataset(filename)
    norm_matrix, ranges, minvals = autonorm(matrix)

    num_rows = matrix.shape[0]
    num_errors = 0.0

    num_test_rows = int(num_rows * hoRatio)
    train_matrix = norm_matrix[num_test_rows:num_rows]
    train_labels = labels[num_test_rows:num_rows]

    for i in range(num_test_rows):
        testvec = norm_matrix[i,:]
        classifier_result = classify0(testvec,
                                      train_matrix, train_labels, 
                                      3)
        actual_result = labels[i]
        if classifier_result != actual_result:
            num_errors += 1.0
     
    return (num_errors/float(num_test_rows))


def img2vector(filename):
    vector = np.zeros((1,1024))
    f = open(filename)
    for i in range(32):
        l = f.readline()
        for j in range(32):
            vector[0,32*i + j] = int(l[j])
    return vector


def handwritingLoadTrain(traindir):
    """
    Load trainingDigits files
    """
    labels = []
    trainfiles = glob.glob('%s/*.txt' % traindir)
    m = len(trainfiles)
    train_matrix = np.zeros((m, 1024))
    for i in range(m):
        thisfile = trainfiles[i]
        thislabel = int(basename(thisfile).split('_')[0])
        labels.append(thislabel)
        train_matrix[i,:] = img2vector(thisfile)

    return train_matrix, labels


def handwritingClassTest(k, traindir = 'data/ch02/trainingDigits', testdir = 'data/ch02/testDigits'):

    train_matrix, labels = handwritingLoadTrain(traindir)

    testfiles = glob.glob('%s/*.txt' % testdir)
    mtest = len(testfiles)
    errorCount = 0

    for i in range(mtest):
        thisfile = testfiles[i]
        actualClass = int(basename(thisfile).split('_')[0])
        testvec = img2vector(thisfile)

        predictedClass = classify0(testvec,
                                   train_matrix, labels, k)
        if actualClass != predictedClass:
            errorCount += 1
            args = (basename(thisfile), actualClass, predictedClass)
            print('%s: class sb %d is %d' % args)
    
    return errorCount, float(errorCount)/float(mtest)

