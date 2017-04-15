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

