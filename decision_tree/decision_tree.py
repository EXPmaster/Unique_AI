from DataLoader import dataset, label
import numpy as np
import pandas as pd

"""
dataset = np.array(dataset)
label = np.array(label)
"""


class Tree:
    def __init__(self, feature=None, value=None, left=None, right=None, data=None, cls=None, to_bottom=False):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.data = data
        self.cls = cls
        self.to_bottom = to_bottom


# CART
class DecisionTree:
    def __init__(self, trainset):
        self.trainset = trainset
        # self.weight = np.ones((len(self.trainset), trainset.shape[1]))
        # self.feature = self.trainset.columns.values.tolist()
        # self.trainset = pd.concat([trainset, label], axis=1)
        self.setnum = 50
        self.setgini = 0.01

    def split_data(self, trainset, feature, featureval):
        dataset1 = trainset[trainset[feature] > featureval]
        dataset2 = trainset[trainset[feature] <= featureval]
        return dataset1, dataset2

    def Gini(self, dataset1, dataset2):
        D1 = len(dataset1)
        D2 = len(dataset2)
        D = len(self.trainset)
        C11 = dataset1[dataset1['Outcome'] == 0].shape[0]
        C12 = dataset1[dataset1['Outcome'] == 1].shape[0]
        C21 = dataset2[dataset2['Outcome'] == 0].shape[0]
        C22 = dataset2[dataset2['Outcome'] == 1].shape[0]

        Gini1 = 1 - (C11 / D1) ** 2 - (C12 / D1) ** 2
        Gini2 = 1 - (C21 / D2) ** 2 - (C22 / D2) ** 2
        Gini = D1 / D * Gini1 + D2 / D * Gini2
        return Gini

    def create_tree(self, trainset, featurevec):
        best_gini = 99
        bestset = None
        bestfeature = None
        bestval = None
        set1 = []
        set2 = []
        for item in featurevec:
            array = []
            for featurevalue in trainset[item]:
                array.append(featurevalue)
            array.sort()
            array = list(set(array))
            for i in range(len(array) - 1):
                featurevalue = (array[i] + array[i + 1]) / 2
                set1, set2 = self.split_data(trainset, item, featurevalue)
                gini = self.Gini(set1, set2)
                if gini < best_gini:
                    bestset = (set1, set2)
                    best_gini = gini
                    bestfeature = item
                    bestval = featurevalue
            """
            print(best_gini)
            print(bestfeature)
            print(bestval)
            print()
            """
        if len(trainset) <= self.setnum or best_gini < self.setgini:
            # print(trainset)
            cls = self.classify(trainset)
            return Tree(data=trainset, cls=cls, to_bottom=True)
        else:
            """
            featurevec.remove(bestfeature)
            new_feature = featurevec
            featurevec.append(bestfeature)
            """
            left = self.create_tree(bestset[0], featurevec)
            right = self.create_tree(bestset[1], featurevec)
            return Tree(
                feature=bestfeature,
                value=bestval,
                left=left,
                right=right,
                data=trainset)

    def classify(self, dataset):
        positive = dataset[dataset['Outcome'] == 1].shape[0]
        negative = dataset[dataset['Outcome'] == 0].shape[0]
        if positive >= negative:
            return 1
        else:
            return 0

    def predict(self, tree, sample):
        current = tree
        while not current.to_bottom:
            if sample[current.feature] > current.value:
                current = current.left
            else:
                current = current.right
        return current.cls

    def accuracy(self, decTree, validset):
        real = 0
        for i in range(len(validset)):
            item_class = self.predict(decTree, validset.iloc[i])
            if item_class == validset.iloc[i]['Outcome']:
                real += 1
        precision = real / len(validset)
        return precision

    def backcut(self, tree, validset):
        accuracy = self.accuracy(tree, validset)
        S = []
        Q = [tree]
        while len(Q):
            current = Q.pop(0)
            S.append(current)
            if not current.left.to_bottom:
                Q.append(current.left)
            if not current.right.to_bottom:
                Q.append(current.right)

        while len(S):
            node = S.pop(-1)
            node.to_bottom = True
            node.cls = self.classify(node.data)
            cur_accuracy = self.accuracy(tree, validset)
            if cur_accuracy >= accuracy:
                accuracy = cur_accuracy
            else:
                node.to_bottom = False
                node.cls = None


if __name__ == '__main__':
    print('loading data...')
    feature = dataset.columns.values.tolist()
    dataset = pd.concat([dataset, label], axis=1)
    trainset = dataset[:500]
    validset = dataset[500:]

    tree = DecisionTree(trainset)
    print('training...')
    decTree = tree.create_tree(trainset, feature)
    print('precision: %lf', tree.accuracy(decTree, validset))
    tree.backcut(decTree, validset)
    print('precision: %lf', tree.accuracy(decTree, validset))
