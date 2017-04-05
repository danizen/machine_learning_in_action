import matplotlib
import matplotlib.pyplot as plt
import numpy as np


decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plotNode(label, centerPt, parentPt, nodetype):
    createPlot.ax1.annotate(label, xy=parentPt, xycoords='axes fraction', 
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodetype, arrowprops=arrow_args)


def retrieveTree(i):
    listOfTrees = [
        { 
            'no surfacing': {
                0: 'no',
                1: {
                    'flippers': {
                        0: 'no',
                        1: 'yes'
                    }
                }
            }
        },
        {
            'no surfacing': {
                0: 'no',
                1: {
                    'flippers': {
                        0: {
                            'head': {
                                0: 'no',
                                1: 'yes'
                            }
                        },
                        1: 'no'
                    }
                }
            }
        }
    ]
    return listOfTrees[i]


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # this is not very duck typed
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
           thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
           thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotMidText(cntrPt, parentPt, label):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, label)


def plotTree(myTree, parentPt, label):
    numLeafs = getNumLeafs(myTree)
    maxDepth = getTreeDepth(myTree)

    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.xOff)
    plotMidText(cntrPt, parentPt, label)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]

    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.xOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(atree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)

    plotTree.totalW = float(getNumLeafs(atree))
    plotTree.totalD = float(getNumLeafs(atree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(atree, (0.5, -0.2), '')
    plt.show()


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]

    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

