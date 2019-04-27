import math


class ConvexHull:
    def __init__(self):
        self.vertices = []
        self.vert_num = 0

    def get_vertice(self, number):
        for i in range(number):
            x = float(input('X axis:'))
            y = float(input('Y axis:'))
            self.vertices.append([x, y])
        self.vert_num = number
        print(self.vertices)

    def __length(self, vec):
        return math.sqrt(vec[0] ** 2 + vec[1] ** 2)

    def __distance(self, i, j):
        vec1 = self.vertices[i]
        vec2 = self.vertices[j]
        return (vec1[0] - vec2[0]) ** 2 + (vec1[1] - vec2[1]) ** 2

    def __angle_fake(self, i, j):
        vec1 = [1, 0]
        vec2 = [self.vertices[j][0] - self.vertices[i][0],
                self.vertices[j][1] - self.vertices[i][1]]
        cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        angle = cross_product / self.__length(vec2)
        # 大于90度, status = 1
        if self.vertices[j][0] < self.vertices[i][0]:
            status = 1
        else:
            status = 0
        return angle, status

    def __direction(self, i, j, k):
        vec1 = []
        vec2 = []
        vec1.append(self.vertices[j][0] - self.vertices[i][0])
        vec1.append(self.vertices[j][1] - self.vertices[i][1])
        vec2.append(self.vertices[k][0] - self.vertices[j][0])
        vec2.append(self.vertices[k][1] - self.vertices[j][1])
        cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        if cross_product > 0:
            # 在左边
            return 1
        else:
            return 0

    def convhull(self):
        if self.vert_num < 3:
            print('No shape')
            return

        y_min_val = self.vertices[0][1]
        x_left_val = self.vertices[0][0]
        index = 0

        # 找到y最小且在最左边的点
        for i in range(1, self.vert_num):
            if self.vertices[i][1] < y_min_val:
                y_min_val = self.vertices[i][1]
                x_left_val = self.vertices[i][0]
                index = i
            elif self.vertices[i][1] == y_min_val:
                if self.vertices[i][0] < x_left_val:
                    x_left_val = self.vertices[i][0]
                    index = i
        # 排序
        angle_acute = []  # 锐角
        index_acute = []
        angle_obtuse = []  # 钝角
        index_obtuse = []
        vertices_sorted = {}
        for i in range(self.vert_num):
            if i == index:
                continue
            a, s = self.__angle_fake(index, i)
            if s == 1:
                if len(angle_obtuse) and a == angle_obtuse[-1]:
                    if self.__distance(i, index) > self.__distance(index_obtuse[-1], index):
                        index_obtuse.pop()
                        angle_obtuse.pop()
                    else:
                        continue
                angle_obtuse.append(a)
                index_obtuse.append(i)
            else:
                if len(angle_acute) and a == angle_acute[-1]:
                    if self.__distance(i, index) > self.__distance(index_acute[-1], index):
                        index_acute.pop()
                        angle_acute.pop()
                    else:
                        continue
                angle_acute.append(a)
                index_acute.append(i)
        vertices_acute = dict(zip(index_acute, angle_acute))
        vertices_obtuse = dict(zip(index_obtuse, angle_obtuse))
        vertices_acute = sorted(vertices_acute.items(), key=lambda item: item[1])
        vertices_obtuse = sorted(vertices_obtuse.items(), key=lambda item: item[1], reverse=True)
        vertices_sorted.update(vertices_acute)
        vertices_sorted.update(vertices_obtuse)
        # 开始操作
        S = []
        vertices_sorted_index = []
        for item in vertices_sorted.keys():
            vertices_sorted_index.append(item)
        S.append(index)
        S.append(vertices_sorted_index[0])
        S.append(vertices_sorted_index[1])
        for i in range(2, len(vertices_sorted_index)):
            while self.__direction(S[-2], S[-1], vertices_sorted_index[i]) == 0:
                S.pop()
            S.append(vertices_sorted_index[i])

        """
        while len(S):
            node = S.pop()
            print(self.vertices[node])
        """
        return S

    def area(self):
        Q = self.convhull()
        point_basis = self.vertices[Q.pop(0)]
        vectors = []
        while len(Q):
            point = self.vertices[Q.pop(0)]
            vectors.append([point[0] - point_basis[0], point[1] - point_basis[1]])

        last_vec = vectors.pop(0)
        areas = []
        while len(vectors):
            this_vec = vectors.pop(0)
            cross_product = last_vec[0] * this_vec[1] - last_vec[1] * this_vec[0]
            areas.append(cross_product / 2)
            last_vec = this_vec

        size = 0
        while len(areas):
            size += areas.pop(0)

        print(size)


C = ConvexHull()
C.get_vertice(5)
C.area()
# C.convhull()
