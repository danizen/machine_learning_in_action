import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loadDataSet(filename):
    mat = np.loadtxt(filename, dtype=np.float32)
    m, n = mat.shape
    dataMat = np.hstack((np.ones((m, 1), dtype=np.float32), np.delete(mat, 2, axis=1)))
    labelMat = np.delete(mat, [0,1], axis=1)
    return dataMat, labelMat


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(- x))


def gradAscent(datamat, labelmat, alpha = 0.001, numcycles = 500):
    m, n = datamat.shape
    weights = np.ones((n, 1))
    for k in range(numcycles):
        h = sigmoid(np.matmul(datamat, weights))
        error = (labelmat - h)
        weights += alpha * np.matmul(datamat.T, error)
    return weights, error.mean()


def stocGradAscent(datamat, labelmat, numcycles = 150):
    m, n = datamat.shape
    weights = np.ones(n)
    for j in range(numcycles):
        dataindex = [i for i in range(m)]
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randindex = int(np.random.uniform(0, len(dataindex)))
            h = sigmoid(sum(datamat[randindex] * weights))
            error = labelmat[randindex] - h
            weights += alpha * error * datamat[randindex]
            del(dataindex[randindex])
    return weights


def plotBestFit(datamat, labelmat, weights):
    n = datamat.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xred = []; yred = []
    xgreen = []; ygreen = []
    for i in range(n):
        if int(labelmat[i]) == 1:
            xred.append(datamat[i, 1])
            yred.append(datamat[i, 2])
        else:
            xgreen.append(datamat[i, 1])
            ygreen.append(datamat[i, 2])
    ax.scatter(xred, yred, s=30, c='red', marker='s')
    ax.scatter(xgreen, ygreen, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

