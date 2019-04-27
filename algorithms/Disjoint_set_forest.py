class Tree:
    def __init__(self, item):
        self.item = item
        self.parent = None
        self.rank = 0


class DisjointForest:
    def __init__(self):
        self.sets = []
        self.sets_num = 0

    def make_set(self, item):
        node = Tree(item)
        node.rank = 0
        node.parent = node
        self.sets_num += 1
        return node

    def find_set(self, node):
        if node != node.parent:
            node.parent = self.find_set(node.parent)
        return node.parent

    def __link(self, x, y):
        if x.rank > y.rank:
            y.parent = x
        else:
            x.parent = y
            if x.rank == y.rank:
                y.rank += 1

    def union(self, x, y):
        self.__link(self.find_set(x), self.find_set(y))
        self.sets_num -= 1
