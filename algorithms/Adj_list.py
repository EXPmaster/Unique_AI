from Disjoint_set import DisjointSet
from Disjoint_set_forest import DisjointForest


class MyGraph:
    # 邻接表
    def __init__(self):
        # test data
        # {0: [1, 2], 1: [0, 3, 4], 2: [0, 5, 6], 3: [1, 7], 4: [1, 7], 5: [2, 6], 6: [2, 5], 7: [3, 4]}
        # num = 8
        self.graph = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: [1], 4: [
            5, 6], 5: [4], 6: [4, 7], 7: [6, 8], 8: [7], 9: []}
        self.node_num = 10
        self.visited = []
        self.indeg = []

    def init_graph(self, number):
        self.graph = {i: [] for i in range(number)}
        self.node_num = number
        for key in self.graph.keys():
            while True:
                temp = input('Enter neighbors of node ' +
                             str(key + 1) + ', # to end')
                if temp == '#':
                    break
                else:
                    self.graph[key].append(int(temp) - 1)
        print(self.graph)

    def DFS(self):
        self.visited = []
        for i in range(self.node_num):
            self.visited.append(False)
        for i in range(self.node_num):
            if not self.visited[i]:
                self.__DFS_visit(i)

    def __DFS_visit(self, v):
        self.visited[v] = True
        print(v + 1)
        for w in self.graph[v]:
            if not self.visited[w]:
                self.__DFS_visit(w)

    def BFS(self):
        self.visited = []
        for i in range(self.node_num):
            self.visited.append(False)
        Q = []
        for v in range(self.node_num):
            if not self.visited[v]:
                self.visited[v] = True
                print(v + 1)
                Q.append(v)
                while len(Q):
                    node = Q.pop(0)
                    for w in self.graph[node]:
                        if not self.visited[w]:
                            self.visited[w] = True
                            print(w + 1)
                            Q.append(w)

    def topo_sort(self):
        for i in range(self.node_num):
            self.indeg.append(0)
        for i in self.graph.keys():
            for j in self.graph[i]:
                self.indeg[j] += 1
        S = []
        for i in range(self.node_num):
            if not self.indeg[i]:
                S.append(i)
        count = 0
        while len(S):
            node = S.pop()
            print(node + 1)
            count += 1
            for j in self.graph[node]:
                self.indeg[j] -= 1
                if not self.indeg[j]:
                    S.append(j)

        if count < self.node_num:
            print('The graph has circuits')

    # 连通分量
    def connected_component(self, S):
        self.visited = []
        for i in range(self.node_num):
            S.make_set(i)
            self.visited.append(False)
        for i in range(self.node_num):
            self.visited[i] = True
            for item in self.graph[i]:
                if not self.visited[item]:
                    if S.find_set(i) != S.find_set(item):
                        S.union(i, item)
        S.traverse()

    def connected_component_optimize(self, S):
        sets = []
        self.visited = []
        for i in range(self.node_num):
            sets.append(S.make_set(i))
            self.visited.append(False)
        for i in sets:
            self.visited[i.item] = True
            for item in self.graph[i.item]:
                if not self.visited[item]:
                    if S.find_set(i) != S.find_set(sets[item]):
                        S.union(i, sets[item])
        for item in sets:
            print(item.item, item.parent.item)


G = MyGraph()
S = DisjointForest()
# G.init_graph(10)
G.connected_component_optimize(S)

S2 = DisjointSet()
G.connected_component(S2)
