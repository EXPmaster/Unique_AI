class config(object):
    def __init__(self):
        self.TRAIN_EXAMPLE = './train-images-idx3-ubyte/train-images.idx3-ubyte'
        self.TRAIN_LABEL = './train-labels-idx1-ubyte/train-labels.idx1-ubyte'
        self.TEST_EXAMPLE = './t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
        self.TEST_LABEL = './t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'
        self.MODEL_PATH = './model.t7'
        self.EPOCH = 10
        self.LR = 1e-4
        self.TRAIN_BATCH_SIZE = 128
        self.TEST_BATCH_SIZE = 128
        self.BATCH_SIZE = 100
        self.INPUT_SIZE = 28 * 28
        self.HIDDEN_SIZE = 100
        self.NUM_CLASS = 10


