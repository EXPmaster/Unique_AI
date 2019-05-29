from node import *
from dataloader import Mydataset
import copy


class Node:
    def __init__(self, item):
        self.item = item
        self.gradient = 0
        self.left = None
        self.right = None
        self.parent = None
        self.operator = None


class CP:
    def __init__(self, x):
        self.data = x
        self.operator = None
        self.gradient = 0

    def __add__(self, other):
        self.operator = ReLUAdd()
        result = self.operator.forward(self.data.item, other.data.item)
        node = Node(result)
        node.operator = self.operator
        self.data.parent = node
        node.left = self.data
        node.right = other.data
        result = CP(node)
        return result

    def __mul__(self, other):
        self.operator = MultiplyGate()
        result = self.operator.forward(self.data.item, other.data.item)
        node = Node(result)
        node.operator = self.operator
        self.data.parent = node
        node.left = self.data
        node.right = other.data
        result = CP(node)
        return result

    def __truediv__(self, other):
        self.operator = DivGate()
        result = self.operator.forward(self.data.item, other.data.item)
        node = Node(result)
        node.operator = self.operator
        self.data.parent = node
        node.left = self.data
        node.right = other.data
        result = CP(node)
        return result


class NeuronNetwork:
    def __init__(self, dim_in, dim_hid, dim_out, hid_size=1):
        self.result = None
        self.weights = []
        self.bias = []
        for i in range(hid_size):
            if i == 0:
                weight = np.random.randn(dim_in, dim_hid)
            else:
                weight = np.random.randn(dim_hid, dim_hid)
            self.weights.append(CP(Node(weight)))
            self.bias.append(CP(Node(1)))
        w = np.random.randn(dim_hid, dim_out)
        self.bias.append(CP(Node(1)))
        self.weights.append(CP(Node(w)))

    def forward(self, dataset, label):
        temp = CP(Node(dataset))
        bias = CP(Node(0))
        for (i, w) in enumerate(self.weights):
            last_temp = temp
            if i != len(self.weights) - 1:
                temp = temp * w + bias
            else:
                temp = temp * w
        loss = np.square(temp.data.item - label).sum()
        # print(loss)
        self.result = temp.data
        return 2.0 * (temp.data.item - label), self.result.item

    def forward_t(self, x, y):
        z = x * y
        # w = copy.deepcopy(z)
        # w.data.item = w.data.item.T
        # ans = w * z
        self.result = z.data
        return 2.0 * z.data.item

    def backward(self, loss):
        dz = loss
        self.__preorder(self.result, dz)

    def __preorder(self, current, gradient):
        if current:
            current.gradient = gradient

            if current.operator:
                dx, dy = current.operator.backward(gradient)

            else:
                dx = dy = None
            # print(dx, dy)
            self.__preorder(current.left, dx)
            self.__preorder(current.right, dy)

    def update(self):
        lr = 1e-6
        for item in self.weights:
            item.data.item -= lr * item.data.gradient

        for item in self.bias:
            item.data.item -= lr * item.data.gradient


# data, label = np.random.randn(64, 1000), np.random.randn(64, 10)
def model(model, G):
    total = 0
    currect = 0
    for j, data in enumerate(model.test_loader):
        inputs, labels = data
        inputs = inputs.numpy().reshape((100, 28 * 28))
        labels = labels.numpy().reshape(100)
        label = np.zeros((100, 10))
        for index, datas in enumerate(labels):
            label[index][datas] = 1
        _, predicts = G.forward(inputs, label)
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


"""
matrix = np.array([[0.1, 0.5], [-0.3, 0.8]])
x = np.array([[0.2], [0.4]])
matrix = CP(Node(matrix))
x = CP(Node(x))
Graph = NeuronNetwork(dim_in=2, dim_hid=1, dim_out=1, hid_size=1)
loss = Graph.forward_t(matrix, x)
Graph.backward(loss)
"""

if __name__ == '__main__':
    Graph = NeuronNetwork(dim_in=28 * 28, dim_hid=1, dim_out=10, hid_size=1)
    d = Mydataset()
    for i in range(100):
        print('epoch %d' % (i + 1))
        # train
        for j, data in enumerate(d.train_loader):
            inputs, labels = data
            inputs = inputs.numpy().reshape((100, 28 * 28))
            labels = labels.numpy().reshape(100)
            label = np.zeros((100, 10))
            for index, datas in enumerate(labels):
                label[index][datas] = 1

            # print(inputs.shape, labels.shape)

            loss, predicts = Graph.forward(inputs, label)
            if j % 100 == 99:
                print(np.square(loss / 2).sum())
            Graph.backward(loss)
            Graph.update()

        model(d, Graph)
