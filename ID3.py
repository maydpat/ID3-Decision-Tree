##############
# Name: Mayur Patil
# email: patil37@purdue.edu
# Date: March 4, 2019

import pandas as pd
import numpy as np
import sys
import os

def entropy(freqs):
    """
    entropy(p) = -SUM (Pi * log(Pi))
    >>> entropy([10.,10.])
    1.0
    >>> entropy([10.,0.])
    0
    >>> entropy([9.,3.])
    0.811278
    """
    all_freq = sum(freqs)
    entropy = 0
    for fq in freqs:
        prob = fq / (1.0 * all_freq)
        if abs(prob) > 1e-8:
            entropy += -prob * np.log2(prob)
    return entropy

def infor_gain(before_split_freqs, after_split_freqs):
    """
    gain(D, A) = entropy(D) - SUM ( |Di| / |D| * entropy(Di) )
    >>> infor_gain([9,5], [[2,2],[4,2],[3,1]])
    0.02922
    """
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    for freq in after_split_freqs:
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain

def load_data(fileData, fileLabels):
    fileData = pd.read_csv(fileData, delimiter = ',',  index_col=None, engine='python')
    fileLabels = pd.read_csv(fileLabels, delimiter = ',',  index_col=None, engine='python')
    fileData = pd.concat([fileData, fileLabels], axis=1)
    return fileData

def countTotalNodes(rootNode):
    pass

def isTree(node):
    if node != None:
        if node.leftNode != None or node.rightNode != None:
            return True
        else:
            return False

nodes = []

def find_best_split(dataset, columns):
    gain = 0
    value = 0
    rightNode = None
    leftNode = None
    df1Final = 0
    df2Final = 0
    splitProperty = 'none'
    for col in columns:
        if col == 'survived':
            continue
        columnDataPD = dataset[col]
        columnDataOriginal = []
        for row in columnDataPD:
            columnDataOriginal.append(row)
        columnDataOriginal.sort()
        columnDataNoDupes = list(set(columnDataOriginal))
        for index in range(len(columnDataNoDupes)-1):
            midpoint = (columnDataNoDupes[index] + columnDataNoDupes[index+1])/2
            df1 = dataset[dataset[col] <= midpoint]
            df2 = dataset[dataset[col] > midpoint]
            df1Survived = df1[df1['survived'] > 0].values
            df1Died = df1[df1['survived'] == 0].values
            df2Survived = df2[df2['survived'] == 1].values
            df2Died = df2[df2['survived'] == 0].values
            datasetSurvived = dataset[dataset['survived'] == 1].values
            datasetDied = dataset[dataset['survived'] == 0].values
            tempGain = infor_gain([len(datasetSurvived), len(datasetDied)], [[len(df1Survived), len(df1Died)], [len(df2Survived), len(df2Died)]])
            if tempGain > gain:
                gain = tempGain
                value = midpoint
                splitProperty = col
                df1Final = df1
                df2Final = df2

    newNode = Node(value, gain, leftNode, rightNode, splitProperty, 1, False)
    nodes.append(newNode)
    return newNode, df1Final, df2Final

def build_tree(dataset, columns, currentDepth, maxDepth, min_split):
    #if curr dephth is max depth, make current node a leaf node
    tree, df1, df2 = find_best_split(dataset, columns)
    if currentDepth >= maxDepth or len(dataset) < min_split:
        survived = dataset[dataset['survived'] == 1].values
        died = dataset[dataset['survived'] == 0].values
        tree.terminal = True
        if len(survived) > len(died):
            tree.survived = 1
            tree.splitProperty = 'survived'
        else:
            tree.survived = 0
            tree.splitProperty = 'survived'
        return tree
    if tree.splitProperty == 'none':
        survived = dataset[dataset['survived'] == 1].values
        died = dataset[dataset['survived'] == 0].values
        tree.terminal = True
        if len(survived) > len(died):
            tree.survived = 1
            tree.splitProperty = 'survived'
        else:
            tree.survived = 0
            tree.splitProperty = 'survived'
        return tree
    if tree.gain == 0:
        return tree

    tree.leftNode = build_tree(df1, columns, currentDepth+1, maxDepth, min_split)
    tree.rightNode = build_tree(df2, columns, currentDepth+1, maxDepth, min_split)
    return tree

def predict(tree, row, columns):
    # if row[columns.index(tree.splitProperty)] == tree.value:
    if tree != None and isTree(tree):
        if row[tree.splitProperty] <= tree.value:
            if isTree(tree.leftNode):
                return predict(tree.leftNode, row, columns)
            else:
                return tree.leftNode.survived
        else:
            if isTree(tree.rightNode):
                return predict(tree.rightNode, row, columns)
            else:
                return tree.rightNode.survived

def accuracyMeasure(dataset, tree, columns):
    correct = 0
    index = 0
    for index, row in dataset.iterrows():
        prediction = predict(tree,row,columns)
        if row['survived'] == prediction:
            correct += 1
    return correct / float(len(dataset))

class Node(object):
    def __init__(self, value, gain, leftNode, rightNode, splitProperty, survived, terminal):
        self.value = value
        self.gain = gain
        self.leftNode = None
        self.rightNode = None
        self.splitProperty = splitProperty
        self.survived = 1
        self.terminal = False


def prune(treeHead, treeNode, dataset, columns):
    if not isTree(treeNode.rightNode) and isTree(treeNode) and not isTree(treeNode.leftNode):
        originalAccuracy = accuracyMeasure(dataset, treeHead, columns)
        tempTerminal = treeNode.terminal
        tempSplitProperty = treeNode.splitProperty
        tempLeftNode = treeNode.leftNode
        tempSurvived = treeNode.survived
        tempRightNode = treeNode.rightNode
        treeNode.leftNode = None
        treeNode.rightNode = None
        survived = dataset[dataset['survived'] == 1].values
        died = dataset[dataset['survived'] == 0].values
        treeNode.terminal = True
        numberOfDied = len(died)
        numberOfSurvived = len(survived)
        if numberOfDied < numberOfSurvived:
            treeNode.survived = 1
            treeNode.splitProperty = 'survived'
        else:
            treeNode.survived = 0
            treeNode.splitProperty = 'survived'
        newAccuracy = accuracyMeasure(dataset, treeHead, columns)
        if newAccuracy >= originalAccuracy:
            prune(treeHead, treeHead, dataset, columns)
        else:
            treeNode.leftNode = tempLeftNode
            treeNode.rightNode = tempRightNode
            treeNode.terminal = tempTerminal
            treeNode.splitProperty = tempSplitProperty
            treeNode.survived = tempSurvived
    else:
        if None != treeNode.rightNode:
            prune(treeHead, treeNode.rightNode, dataset,columns)
        if None != treeNode.leftNode:
            prune(treeHead, treeNode.leftNode, dataset,columns)

if __name__ == "__main__":
	# parse arguments
    if 'vanilla' == sys.argv[3]:
        trainDirContents = os.listdir(sys.argv[1])
        testDirContents = os.listdir(sys.argv[2])
        trainDataFile = next(x for x in trainDirContents if '.data' in x)
        trainLabelFile = next(x for x in trainDirContents if '.label' in x)
        testDataFile = next(x for x in testDirContents if '.data' in x)
        testLabelFile = next(x for x in testDirContents if '.label' in x)
        trainDataOriginal = load_data(sys.argv[1]+'/'+trainDataFile, sys.argv[1]+'/'+trainLabelFile)
        testData = load_data(sys.argv[2]+'/'+testDataFile, sys.argv[2]+'/'+testLabelFile)
        trainDataOriginal = trainDataOriginal[0:int(len(trainDataOriginal) * int(sys.argv[4]) / 100)]
        columns = []
        for column in trainDataOriginal:
            columns.append(column)
        tree = build_tree(trainDataOriginal, columns, 1, float("inf"), 0)

        trainAccuracy = accuracyMeasure(trainDataOriginal, tree, columns,)
        testAccuracy = accuracyMeasure(testData, tree, columns)
        print('Train set accuracy: %.4f' % trainAccuracy)
        print('Test set accuracy: %.4f' % testAccuracy)
    if 'depth' == sys.argv[3]:
        trainDirContents = os.listdir(sys.argv[1])
        testDirContents = os.listdir(sys.argv[2])
        trainDataFile = next(x for x in trainDirContents if '.data' in x)
        trainLabelFile = next(x for x in trainDirContents if '.label' in x)
        testDataFile = next(x for x in testDirContents if '.data' in x)
        testLabelFile = next(x for x in testDirContents if '.label' in x)
        trainDataOriginal = load_data(sys.argv[1]+'/'+trainDataFile, sys.argv[1]+'/'+trainLabelFile)
        testData = load_data(sys.argv[2]+'/'+testDataFile, sys.argv[2]+'/'+testLabelFile)
        validation_percent = 100 - int(sys.argv[5])
        maxDepth = int(sys.argv[6])
        trainData = trainDataOriginal[0:int(len(trainDataOriginal) * int(sys.argv[4]) / 100)]
        validationData = trainDataOriginal[int(len(trainDataOriginal) * validation_percent / 100):]
        columns = []
        for column in trainData:
            columns.append(column)
        tree = build_tree(trainData, columns, 1, maxDepth, 0)

        trainAccuracy = accuracyMeasure(trainData, tree, columns)
        validationAccuracy = accuracyMeasure(validationData, tree, columns)
        testAccuracy = accuracyMeasure(testData, tree, columns)
        print('Train set accuracy: %.4f' % trainAccuracy)
        print('Validation set accuracy: %.4f' % validationAccuracy)
        print('Test set accuracy: %.4f' % testAccuracy)
    if 'min_split' == sys.argv[3]:
        trainDirContents = os.listdir(sys.argv[1])
        testDirContents = os.listdir(sys.argv[2])
        trainDataFile = next(x for x in trainDirContents if '.data' in x)
        trainLabelFile = next(x for x in trainDirContents if '.label' in x)
        testDataFile = next(x for x in testDirContents if '.data' in x)
        testLabelFile = next(x for x in testDirContents if '.label' in x)
        trainDataOriginal = load_data(sys.argv[1]+'/'+trainDataFile, sys.argv[1]+'/'+trainLabelFile)
        testData = load_data(sys.argv[2]+'/'+testDataFile, sys.argv[2]+'/'+testLabelFile)
        validation_percent = 100 - int(sys.argv[5])
        minimum_split = int(sys.argv[6])
        trainData = trainDataOriginal[0:int(len(trainDataOriginal) * int(sys.argv[4]) / 100)]
        validationData = trainDataOriginal[int(len(trainDataOriginal) * validation_percent / 100):]
        columns = []
        for column in trainData:
            columns.append(column)
        tree = build_tree(trainData, columns, 1, float("inf"), minimum_split)

        trainAccuracy = accuracyMeasure(trainData, tree, columns)
        validationAccuracy = accuracyMeasure(validationData, tree, columns)
        testAccuracy = accuracyMeasure(testData, tree, columns)
        print('Train set accuracy: %.4f' % trainAccuracy)
        print('Validation set accuracy: %.4f' % validationAccuracy)
        print('Test set accuracy: %.4f' % testAccuracy)
    if 'prune' == sys.argv[3]:
        trainDirContents = os.listdir(sys.argv[1])
        testDirContents = os.listdir(sys.argv[2])
        trainDataFile = next(x for x in trainDirContents if '.data' in x)
        trainLabelFile = next(x for x in trainDirContents if '.label' in x)
        testDataFile = next(x for x in testDirContents if '.data' in x)
        testLabelFile = next(x for x in testDirContents if '.label' in x)
        trainDataOriginal = load_data(sys.argv[1]+'/'+trainDataFile, sys.argv[1]+'/'+trainLabelFile)
        testData = load_data(sys.argv[2]+'/'+testDataFile, sys.argv[2]+'/'+testLabelFile)
        validation_percent = 100 - int(sys.argv[5])
        trainData = trainDataOriginal[0:int(len(trainDataOriginal) * int(sys.argv[4]) / 100)]
        validationData = trainDataOriginal[int(len(trainDataOriginal) * validation_percent / 100):]
        columns = []
        for column in trainData:
            columns.append(column)
        tree = build_tree(trainData, columns, 1, float("inf"), 0)
        prune(tree,tree, validationData, columns)


        trainAccuracy = accuracyMeasure(trainData, tree, columns)
        testAccuracy = accuracyMeasure(testData, tree, columns)
        print('Train set accuracy: %.4f' % trainAccuracy)
        print('Test set accuracy: %.4f' % testAccuracy)


	# build decision tree

	# predict on testing set & evaluate the testing accuracy
