import numpy as np
from DataLoader import train_set, test_data, test_id
import pandas as pd
import matplotlib.pyplot as plt

testvalues = np.c_[np.ones((test_data.shape[0], 1)), test_data]


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))


def cost(hyp, y, m):
    return 1.0 / m * (- np.dot(y.T, np.log(hyp)) -
                      np.dot((1 - y).T, np.log(1 - hyp)))


def classify(X, theta):
    if hypothesis(X, theta) > 0.5:
        return 1
    else:
        return 0

# 留一法
def hold_out(dataset):
    np.random.shuffle(dataset)
    size = int(dataset.shape[0] * 0.8)
    train = dataset[:size, :]
    validation = dataset[-size:, :]
    return train, validation

# 自助法
def bootstrap(dataset):
    x_train_index = np.random.choice(
        len(X), round(len(X) * 0.7), replace=True)
    x_valid_index = np.array(list(set(range(len(X))) - set(x_train_index)))
    trainingSet = dataset[x_train_index]
    validation = dataset[x_valid_index]
    return trainingSet, validation

# 交叉验证
def cross_valid(dataset, k):
    subset = np.array_split(dataset, 10)
    validation = subset[k]
    subset.pop(k)
    trainingSet = subset[0]
    return trainingSet, validation


X = train_set
m = X.shape[0]
X = np.c_[np.ones((m, 1)), X]
train, valid = bootstrap(X)
x_train = train[:, :-1]
m = x_train.shape[0]
x_valid = valid[:, :-1]
n = x_valid.shape[0]
x_train_label = train[:, -1].reshape(m,  1)
x_valid_label = valid[:, -1].reshape(n, 1)

theta = np.ones((x_train.shape[1], 1))


if __name__ == '__main__':
    epoch = 1000
    lr = 0.01
    F = 0
    ep = []
    los = []
    for i in range(1, epoch + 1):
        # ep.append(i)
        hyp = hypothesis(x_train, theta)
        loss = cost(hyp, x_train_label, m)
        print('epoch: %d  loss: %lf' % (i, loss))
        grad = 1. / m * np.dot(x_train.T, hyp - x_train_label)
        theta = theta - lr * grad

        # count accuracy
        real = 0
        judge_real = 0

        # precision
        for i in range(x_valid.shape[0]):
            if classify(x_valid[i, :], theta) == 1:
                judge_real += 1
                if x_valid_label[i] == 1:
                    real += 1
        precision = real / judge_real

        # recall
        r_real = 0
        r_judge_real = 0
        for i in range(x_valid.shape[0]):
            if x_valid_label[i] == 1:
                r_real += 1
                if classify(x_valid[i, :], theta) == 1:
                    r_judge_real += 1
        recall = r_judge_real / r_real

        F1 = 2 * precision * recall / (precision + recall)
        F += F1
        print('precision: %lf' % precision)
        print('recall: %lf' % recall)
        print('F1: %lf' % F1)
        ep.append(precision)
        los.append(recall)
        # los.append(float(loss))
    ep = np.array(ep)
    los = np.array(los)

    print('performance: %lf' % (F / epoch))
    plt.plot(ep, los)
    plt.show()
"""
    # output
    survival = []
    for i in range(testvalues.shape[0]):
        survival.append(classify(testvalues[i, :], theta))

    output = pd.DataFrame({'PassengerId': test_id, 'Survived': survival})
    output.to_csv('submission.csv', index=False)
"""
