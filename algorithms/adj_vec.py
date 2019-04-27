import numpy as np


class MyGraph:
    def __init__(self, number):
        self.node = []
        self.node_num = number
        self.adj_matrix = np.zeros((number, number), dtype=int)
        self.inv_matrix = None
        self.visited = []
        self.post = []

    def init_graph(self):
        for i in range(self.node_num):
            self.node.append(i)
        for i in self.node:
            while 1:
                temp = input('Enter neighbors of node ' + str(i + 1) + ', # to end')
                if temp == '#':
                    break
                else:
                    self.adj_matrix[i, int(temp) - 1] = 1
        self.inv_matrix = self.adj_matrix.T

    def __DFS1(self, v):
        self.visited[v] = 1
        for i in range(self.node_num):
            if self.adj_matrix[v][i] == 1 and self.visited[i] == 0:
                self.__DFS1(i)
        self.post.append(v)

    def __DFS2(self, v):
        self.visited[v] = 1
        print(v + 1)
        for i in range(self.node_num):
            if self.inv_matrix[v][i] == 1 and self.visited[i] == 0:
                self.__DFS2(i)

    def SCC(self):
        for i in range(self.node_num):
            self.visited.append(0)
        for node in self.node:
            if not self.visited[node]:
                self.__DFS1(node)
        print(self.post)

        self.visited = []
        for i in range(self.node_num):
            self.visited.append(0)

        while len(self.post):
            item = self.post.pop()
            if not self.visited[item]:
                self.__DFS2(item)
                print('')


G = MyGraph(8)
G.init_graph()
G.SCC()
