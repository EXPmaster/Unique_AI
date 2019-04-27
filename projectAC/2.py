import numpy as np
import numpy.matlib

N = 100
dim_in = 100
dim_out = 1
dim_hid = 100

# input data
X = np.matlib.randn(N, dim_in)  # 100*1000
# output data
Y = np.matlib.randn(N, dim_out)  # 100*1

# weight

learning_rate = 1e-6
w = []  # parameters

m = len(X)
lamda = 1
discount = 0.7


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def lossfunc(hyp, y, theta):
    # + 1 / 2 / lamda * m * (theta.sum() ** 2)
    return 1 / m * (- np.dot(y.T, np.log(hyp)) - (1 - y).T * np.log(1 - hyp))


print(m)
print('enter the layer of network.')

layer = input()
layer = int(layer)
print('%d layer network' % layer)

for i in range(layer):
    if i == 0:
        w.append(np.matlib.randn(dim_in, dim_hid))
    elif i != layer:
        w.append(np.matlib.randn(dim_hid, dim_hid))
    else:
        w.append(np.matlib.randn(dim_hid, dim_out))

delta = []
for i in range(layer):
    if i == 0:
        delta.append(np.zeros((dim_in, dim_hid)))
    else:
        delta.append(np.zeros(dim_hid))

dt = []
for i in range(layer):
    if i == 0:
        dt.append(np.zeros((dim_in, dim_hid)))
    elif i != layer:
        dt.append(np.zeros(dim_hid))
    else:
        dt.append(np.zeros((dim_hid, dim_out)))

if __name__ == '__main__':
    for epoch in range(100):
        # forward
        z = []
        z.append(X)
        z1 = X.dot(w[0])
        hypothesis_func = sigmoid(z1)
        for i in range(1, layer):
            z.append(hypothesis_func.dot(w[i]))
            hypothesis_func = sigmoid(z[i])

        predict = hypothesis_func.dot(w[layer - 1])
        predict = sigmoid(predict)
        loss = lossfunc(predict, Y, w[layer - 1]).sum()
        print('epoch:%d, loss:%lf' % (epoch, loss))

        # backward
        delta[layer - 1] = predict - Y
        # layer 2 to n-1
        length = layer - 2
        for i in range(0, layer - 1):
            delta[length - i] = w[length - i].T.dot(delta[length - i + 1]) * sigmoid(
                z[length - i]) * (1 - sigmoid(z[length - i]))
            dt[length - i] = dt[length - i] + \
                delta[length - i + 1].dot(sigmoid(z[length - i]).T)
            D = 1 / m * (dt[length - i] + lamda * w[length - i])
            w[length - i] = w[length - i] * discount - D * learning_rate
