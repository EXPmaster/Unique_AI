import numpy as np


class AddGate:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.ans = 0
        self.name = 'AddGate'

    def forward(self, x, y):
        self.x = x
        self.y = y
        self.ans = x + y
        return self.ans

    def backward(self, dz):
        dx = dy = dz
        return dx, dy


class MultiplyGate:
    def __init__(self):
        self.x = None
        self.w = None
        self.ans = None
        self.name = 'MultiplyGate'

    def forward(self, x, y):
        self.x = x
        self.w = y
        self.ans = np.dot(x, y)
        # print(self.ans)
        return self.ans

    def backward(self, dz):
        dx = np.dot(dz, self.w.T)
        dw = np.dot(self.x.T, dz)
        return dx, dw


class SigmoidGate:
    def __init__(self):
        self.input = 0
        self.name = 'SigmoidGate'

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x):
        y = self.sigmoid(x)
        self.input = x
        return y

    def backward(self, dz):
        dx = dz * (self.sigmoid(self.input) * (1 - self.sigmoid(self.input)))
        return dx


class ExpGate:
    def __init__(self):
        self.output = 0
        self.name = 'ExpGate'

    def forward(self, x):
        y = np.exp(x)
        self.output = y
        return y

    def backward(self, dz):
        return dz * self.output


class DivGate:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.name = 'DivGate'

    def forward(self, x, y):
        self.x = x
        self.y = y
        return y / x

    def backward(self, dz):
        dx = - dz * self.y / (self.x ** 2)
        dy = 1.0 / self.x
        return dx, dy


class ReLUAdd:
    def __init__(self):
        self.x = 0
        self.w = 0
        self.ans = 0
        self.name = 'ReLU'

    def relu(self, x):
        s = np.where(x < 0, 0, x)
        return s

    def relu_grad(self, dz):
        grad = np.where(dz > 0, dz, 0)
        return grad

    def forward(self, x, y):
        self.x = x
        self.w = y
        self.ans = self.relu(x + y)
        return self.ans

    def backward(self, dz):
        dx = dy = self.relu_grad(dz)
        return dx, dy
