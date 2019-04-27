import numpy as np
import xlrd
from math import log

file = xlrd.open_workbook("./Tree/data.xlsx")

sheet = file.sheet_by_index(0)
# 用于存储数据的数组

my_list = []
label = []

# 数据预处理
for i in range(1, 252):
    temp = []
    for j in range(9):
        temp.append(sheet.cell(i, j).value.strip('\''))
    my_list.append(temp)
    if sheet.cell(i, 9).value.strip('\'') == 'no-recurrence-events':
        label.append([0])
    else:
        label.append([1])

Mydataset = np.array(my_list)
Mylabel = np.array(label)

array = []
for i in range(len(Mydataset[0, :])):
    idx1 = sorted(set(Mydataset[:, i]))
    idx1 = {word: k for k, word in enumerate(idx1)}
    temp = []
    for item in Mydataset[:, i]:
        for key in idx1:
            if item == key:
                temp.append(idx1[key])
    array.append(temp)

Mydataset = np.array(array).T


# print(Mydataset.shape)


class DecisionTree:
    def __init__(self, threshold):
        self.Dataset = Mydataset
        self.Label = Mylabel
        self.featureVal = {}
        self.threshold = threshold
        self.tree = self.createTree(
            range(
                0, len(
                    self.Dataset)), range(
                0, len(
                    self.Dataset[0])))

        for data in self.Dataset:
            for index, value in enumerate(data):
                if index not in self.featureVal.keys():
                    self.featureVal[index] = [value]
                if value not in self.featureVal[index]:
                    self.featureVal[index].append(value)

    def count_label(self, dataset):
        label_count = {}
        for i in dataset:
            if self.Label[i] in label_count.keys():
                label_count[self.Label[i]] += 1
            else:
                label_count[self.Label[i]] = 1
        return label_count

    # 计算信息熵
    def caculEntropy(self, dataset):
        label_count = self.count_label(dataset)
        size = len(dataset)
        result = 0
        for i in label_count.values():
            pi = i / float(size)
            result -= pi * (log(pi) / log(2))
        return result

    # 划分数据集
    def split_dataset(self, dataset, feature, value):
        result = []
        for index in dataset:
            if self.Dataset[index][feature] == value:
                result.append(index)
        return result

    def calcuGain(self, dataset, feature):
        values = self.featureVal[feature]  # 特征所有可能取值
        result = 0
        for v in values:
            subset = self.split_dataset(
                dataset=dataset, feature=feature, value=v)
            result += len(subset) / float(len(dataset)) * \
                self.caculEntropy(subset)
        return self.caculEntropy(dataset=dataset) - result

    def createTree(self, dataset, features):
        labelCount = self.count_label(dataset)
        # 如果特征集为空，则该树为单节点树
        # 计算数据集中出现次数最多的标签
        if not features:
            return max(list(labelCount.items()), key=lambda x: x[1])[0]

        # 如果数据集中，只包同一种标签，则该树为单节点树
        if len(labelCount) == 1:
            return labelCount.keys()[0]

        # 计算特征集中每个特征的信息增益
        l = map(
            lambda x: [
                x,
                self.calcuGain(
                    dataset=dataset,
                    feature=x)],
            features)

        # 选取信息增益最大的特征
        feature, gain = max(l, key=lambda x: x[1])

        # 如果最大信息增益小于阈值，则该树为单节点树
        if self.threshold > gain:
            return max(list(labelCount.items()), key=lambda x: x[1])[0]

        tree = {}
        # 选取特征子集
        subFeatures = filter(lambda x: x != feature, features)
        tree['feature'] = feature
        # 构建子树
        for value in self.featureVal[feature]:
            subDataset = self.split_dataset(
                dataset=dataset, feature=feature, value=value)

            # 保证子数据集非空
            if not subDataset:
                continue
            tree[value] = self.createTree(
                dataset=subDataset, features=subFeatures)
        return tree

    def f(self, tree, data):
        if not isinstance(tree, dict):
            return tree
        else:
            return self.f(tree[data[tree['feature']]], data)


tree = DecisionTree(threshold=0)
print(tree.tree)
