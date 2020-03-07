import numpy as np
from sklearn import datasets


class Dbscan:

    def __init__(self, dataset, target, epsilon, minPoint):
        self.dataset = dataset
        self.target = target
        self.epsilon = epsilon
        self.minPoint = minPoint
        self.cluster = []
    
    def _find_near_neighbor(self, datapoint):
        diff = (datapoint-self.dataset)**2
        diff_dis = np.sqrt(diff.sum(axis=1))
        data_idx = list(np.where(diff_dis<=self.epsilon)[0])
        # distance_dict = dict(zip([i for i in range(len(self.dataset))], diff_dis))
        return set(data_idx)

    def train(self):
        omega = set()
        for i, item in enumerate(range(len(self.dataset))):
            neighbor = self._find_near_neighbor(self.dataset[item])
            if len(neighbor) >= self.minPoint:
                omega = omega | set([i])
        cluster = []
        k = 0
        not_visited = set([i for i in range(len(self.dataset))])
        while len(omega):
            old_not_visited = not_visited.copy()
            rand_idx = np.random.randint(0, len(omega))
            q = [list(omega)[rand_idx]]
            not_visited -= set(q)
            
            while len(q):
                item_idx = q.pop(0)
                neighbor = self._find_near_neighbor(self.dataset[item_idx])
                if len(neighbor) >= self.minPoint:
                    delta = neighbor & not_visited
                    q += list(delta)
                    not_visited -= delta
                    
            k += 1
            datapoints = old_not_visited - not_visited
            omega -= datapoints
            cluster.append(list(datapoints))
        self.cluster = cluster

    def view(self):
        for idxs in self.cluster:
            label_set = self.target[idxs]
            print(len(label_set))
            print(label_set)
            print()

        
if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_target = iris.target
    dbscan = Dbscan(iris_data, iris_target, 0.3, 4)
    dbscan.train()
    dbscan.view()