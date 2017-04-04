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

    pass


def createPlot(atree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    #axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5, 1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    
    
    #plotTree.totalW = float(getNumLeafs(atree))
    #plotTree.totalD = float(getNumLeafs(atree))
    #plotTree.xOff = -0.5/plotTree.totalW
    #plotTree.yOff = 1.0
    #plotTree(atree, (0.5, 1.0), '')
    plt.show()

