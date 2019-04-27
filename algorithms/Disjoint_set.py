class Node:
    def __init__(self, item, p=None, q=None):
        self.item = item
        self.next = p
        self.front = q


class DisjointSet:
    def __init__(self):
        self.size = 0
        self.heads = []
        self.tails = []

    def make_set(self, item):
        # item = int(input('enter a number:'))
        node = Node(item)
        self.heads.append(node)
        self.tails.append(node)
        node.front = node
        self.size += 1

    def union(self, set1, set2):
        self.tails[set1].next = self.heads[set2]
        self.tails[set1] = self.tails[set2]
        self.size -= 1
        current = self.heads[set2]
        while current:
            current.front = self.heads[set1]
            current = current.next
        self.heads[set2] = self.heads[set1]
        # self.tails[set2] = self.tails[set1]
        # self.heads.pop(set2)
        # self.tails.pop(set2)
        for i in range(self.size):
            if self.heads[i] == self.heads[set1]:
                self.tails[i] = self.tails[set1]

    def traverse(self):
        temp = '#'
        for element in self.heads:
            if temp != element.item:
                current = element
                while current:
                    print(current.item + 1, end='')
                    current = current.next
                print(' ', end='')
                temp = element.item

    def find_set(self, x):
        for element in self.heads:
            current = element
            while current:
                if current.item == x:
                    return current.front
                current = current.next
        print('Cannot find the item')

