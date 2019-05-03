import numpy as np
import pandas as pd
from DataLoader import dataset, label
from decision_tree import DecisionTree
import math


def bootstrap(dataset):
    x_train_index = np.random.choice(
        len(dataset), round(len(dataset)), replace=True)
    x_valid_index = np.array(list(set(range(len(dataset))) - set(x_train_index)))
    trainingSet = dataset.iloc[x_train_index]
    validation = dataset.iloc[x_valid_index]
    return trainingSet, validation


class RandomForest:
    def __init__(self, k, tree_num, feature):
        self.feature_num = k
        self.tree_num = tree_num
        self.forest = []
        self.forest_class = []
        self.feature = feature

    def train(self, dataset):
        for i in range(self.tree_num):
            index_feature = np.random.choice(
                len(self.feature), round(self.feature_num), replace=False)
            print(index_feature)
            featurevec = []
            for item in index_feature:
                featurevec.append(self.feature[item])
            trainset, validset = bootstrap(dataset)
            tree = DecisionTree(trainset)
            dctree = tree.create_tree(trainset, featurevec)
            tree.backcut(dctree, validset)
            self.forest.append(dctree)
            self.forest_class.append(tree)

    def classify(self, validset):
        cls = []
        for i in range(len(self.forest)):
            cls.append(self.forest_class[i].predict(self.forest[i], validset))
        m = cls.count(1)
        if m >= len(cls) / 2:
            return 1
        else:
            return 0

    def accuracy(self, validset):
        real = 0
        for i in range(len(validset)):
            item_class = self.classify(validset.iloc[i])
            if item_class == validset.iloc[i]['Outcome']:
                real += 1
        precision = real / len(validset)
        return precision


if __name__ == '__main__':
    print('loading data...')
    feature = dataset.columns.values.tolist()
    k = int(math.log2(len(feature)))
    # k = 4
    tree_num = 10
    dataset = pd.concat([dataset, label], axis=1)
    trainset = dataset[:500]
    validset = dataset[500:]
    forest = RandomForest(k, tree_num, feature)
    forest.train(trainset)
    print(forest.accuracy(validset))
