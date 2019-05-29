import numpy as np
from dataloader import Mydataset

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 100, 28*28, 10, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)
d = Mydataset()

learning_rate = 1e-5
for t in range(500):
    # Forward pass: compute predicted y
    for j, data in enumerate(d.train_loader):
        inputs, labels = data
        inputs = inputs.numpy().reshape((100, 28 * 28))
        labels = labels.numpy().reshape(100)
        label = np.zeros((100, 10))
        for index, datas in enumerate(labels):
            label[index][datas] = 1
        x = inputs
        y = label
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        if j % 100 == 99:
            print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

        total = 0
        currect = 0
        for j, data in enumerate(d.test_loader):
            inputs, labels = data
            inputs = inputs.numpy().reshape((100, 28 * 28))
            labels = labels.numpy().reshape(100)
            label = np.zeros((100, 10))
            for index, datas in enumerate(labels):
                label[index][datas] = 1
            x = inputs
            y = label
            h = x.dot(w1)
            h_relu = np.maximum(h, 0)
            predicts = h_relu.dot(w2)
            ans = np.zeros(100)
            for i in range(100):
                max = 0
                index = 0
                for k in range(10):
                    if predicts[i][k] > max:
                        max = predicts[i][k]
                        index = k
                ans[i] = index

            total += 100
            currect += (ans == labels).sum()
            # print(ans.shape, labels.shape)
        print('accuracy: %.2f' % (currect / total))