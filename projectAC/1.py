import numpy as np

mat1 = [[4, 2, 1, 5], [8, 7, 2, 10], [4, 8, 3, 6], [6, 8, 4, 9]]
mat2 = [[2, 4], [4, 3], [2, 2], [3, 5]]
matrix = [[1], [2], [3], [4]]
mat3 = np.array(mat1)


def is_matrix(mat):
    length = len(mat[0])
    for i in range(1, len(mat)):
        if length != len(mat[i]):
            print('not matrix')
            return False
    return True


# judge if two matrix can add
def can_add(mat1, mat2):
    if is_matrix(mat1) and is_matrix(mat2):
        length1_row = len(mat1)
        length2_row = len(mat2)
        length1_col = len(mat1[0])
        length2_col = len(mat2[0])
        if (length1_col == length2_col and
                length1_row == length2_row):
            return True
    print('cannot add')
    return False


# 矩阵加法
def mat_add(mat1, mat2):
    if can_add(mat1, mat2):
        result = []
        for i in range(len(mat1)):  # row
            temp = []
            for j in range(len(mat1[0])):  # column
                temp.append(mat1[i][j] + mat2[i][j])
            result.append(temp)
        return np.array(result)
    else:
        return False


def can_multi(mat1, mat2):
    if is_matrix(mat1) and is_matrix(mat2):
        if len(mat1[0]) == len(mat2):
            return True
    else:
        print('error')
        return False


# 矩阵乘法
def mat_multi(mat1, mat2):
    if can_multi(mat1, mat2):
        row = len(mat1)
        col1 = len(mat1[0])
        col2 = len(mat2[0])
        result = []
        for i in range(row):
            list = []
            for j in range(col2):
                temp = 0
                for k in range(col1):
                    temp += mat1[i][k] * mat2[k][j]
                list.append(temp)
            result.append(list)
        return np.array(result)
    else:
        print('cannot multiply')
        return False


# 计算余子式(矩阵)
def calcu_yuzishi(mat, i, m=0):
    size = len(mat)
    list = []
    for j in range(size):
        temp = []
        if j == i:
            continue
        for k in range(size):
            if k == m:
                continue
            else:
                temp.append(mat[j][k])
        list.append(temp)

    return list


# 计算行列式
def calcu_det(mat, det=0):
    size = len(mat)
    if size == 2:
        det = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]
    else:
        for i in range(size):
            det += (-1.0) ** (i + 2) * mat[i][0] * \
                calcu_det(calcu_yuzishi(mat, i), det)
    return det


# 计算伴随矩阵
def company_mat(mat):
    size = len(mat)
    company = []
    for i in range(size):
        ans = []
        for j in range(size):
            ans.append((-1.0) ** (i + j) * calcu_det(calcu_yuzishi(mat, j, i)))
        company.append(ans)
    return company


# 矩阵求逆
def inv_max(mat):
    row = len(mat)
    col = len(mat[0])
    det = calcu_det(mat)
    if is_matrix(mat) and row == col:
        ans = company_mat(mat)
        for i in range(row):
            for j in range(row):
                ans[i][j] = (1.0 / det * ans[i][j])

        return np.array(ans)
    else:
        print('error')
        return False


# LU 分解
def LU(mat):
    row = len(mat)
    col = len(mat[0])
    if is_matrix(mat) and row == col:
        L = np.eye(col)
        U = np.zeros((col, col))
        for k in range(col):
            # 记录消元系数， l^(-1)=-l
            for i in range(k + 1, col):
                L[i][k] = mat[i][k] / mat[k][k]
            # 记录上三角矩阵U
            for j in range(k, col):
                U[k][j] = mat[k][j]
            # 行初等变换更新mat
            for i in range(k + 1, col):
                for j in range(k + 1, col):
                    mat[i][j] -= L[i][k] * U[k][j]

        return L, U
    else:
        return False


# 解线性方程组
def solve(xishu, ans):
    return mat_multi(inv_max(xishu), ans)


if __name__ == '__main__':
    print(mat_add(mat1, mat2))
    print(mat_multi(mat1, mat2))
    print(inv_max(mat1))
    print(LU(mat1))
    print(solve(mat1, matrix))
