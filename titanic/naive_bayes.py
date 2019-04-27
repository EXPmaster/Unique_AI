import numpy as np
import pandas as pd
from DataLoader import train_set, test_data, test_id
from logi_reg import bootstrap


class NaiveBayes:
    def __init__(self, dataset):
        self.lamda = 1
        self.trainset, self.validset = bootstrap(dataset)
        self.train_num = self.trainset.shape[0]
        self.valid_num = self.validset.shape[0]
        self.true_sample = (self.trainset[:, -1] == 1).sum()
        self.neg_sample = self.train_num - self.true_sample
        self.ptrue = (self.true_sample + self.lamda) / \
            (self.train_num + 2 * self.lamda)
        self.pneg = 1 - self.ptrue
        self.feature_true = {i: {} for i in range(self.trainset.shape[1] - 1)}
        self.feature_neg = {i: {} for i in range(self.trainset.shape[1] - 1)}

        # 计数
        for i in range(self.trainset.shape[1] - 1):
            for j in range(self.train_num):
                if self.trainset[j][-1] == 1:
                    if self.trainset[j][i] not in self.feature_true[i].keys():
                        self.feature_true[i][self.trainset[j][i]] = 1
                    else:
                        self.feature_true[i][self.trainset[j][i]] += 1
                elif self.trainset[j][-1] == 0:
                    if self.trainset[j][i] not in self.feature_neg[i].keys():
                        self.feature_neg[i][self.trainset[j][i]] = 1
                    else:
                        self.feature_neg[i][self.trainset[j][i]] += 1

    def __calcu_prob(self):
        for i in range(self.trainset.shape[1] - 1):
            for j in self.feature_true[i].keys():
                self.feature_true[i][j] = (self.feature_true[i][j] + self.lamda) / (
                    self.true_sample + self.lamda * len(self.feature_true[i]))
            for j in self.feature_neg[i].keys():
                self.feature_neg[i][j] = (self.feature_neg[i][j] + self.lamda) / (
                    self.neg_sample + self.lamda * len(self.feature_neg[i]))

    def classify(self, dataset):
        class_true, class_neg = np.log(self.ptrue), np.log(self.pneg)
        for i in range(dataset.shape[0] - 1):
            if dataset[i] not in self.feature_true[i].keys() \
                    or dataset[i] not in self.feature_neg[i].keys():
                continue
            class_true += np.log(self.feature_true[i][dataset[i]])
            class_neg += np.log(self.feature_neg[i][dataset[i]])
        if class_true > class_neg:
            return 1
        else:
            return 0

    def validate(self):
        sum = 0
        # class_true, class_neg = self.ptrue, self.pneg
        for i in range(self.valid_num):
            if self.classify(self.validset[i, :]) == self.validset[i, -1]:
                sum += 1
        accuracy = sum / self.valid_num
        print(accuracy)


def main():
    testvalues = np.c_[np.ones((test_data.shape[0], 1)), test_data]
    bayes = NaiveBayes(train_set)
    bayes.validate()
    # output
    survival = []
    for i in range(testvalues.shape[0]):
        survival.append(bayes.classify(testvalues[i, :]))

    output = pd.DataFrame({'PassengerId': test_id, 'Survived': survival})
    output.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
