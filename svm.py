import numpy as np


def loadDataSet(filename, dtype=np.float32):
    mat = np.loadtxt(filename, dtype=dtype)
    m, n = mat.shape
    lastcol = n -1 
    datamat = np.delete(mat, lastcol, axis=1)
    labelmat = np.delete(mat, range(0, lastcol), axis=1)
    return datamat, labelmat


def selectJrand(i, m):
    '''
    Select a random number from 0 to m-1 that is not i
    '''
    j=i
    while (j==i): 
        j = int(np.random.uniform(0,m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(data, labels, C, tolerance, maxiter):
    b = 0
    m, n = data.shape
    alphas = np.zeros((m,1))    # to force choice of matrix multiplier
    count = 0
    while count < maxiter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = np.multiply(alphas, labels).T
            pass
        count += 1
    # Not fininshed yet
    return alphas

