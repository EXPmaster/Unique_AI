import pandas as pd
import numpy as np

train_data = pd.read_csv('diabetes.csv')
k = np.ones((5, 704))
# df.loc[df.a>=2,'b'] = 'new_data'
print(train_data)
# train_data.loc[train_data.Insulin==0,'Insulin']=train_data['Insulin'].mean()
Train_data = train_data.iloc[0:500, :]
Test_data = train_data.iloc[500:769]
Test_data.loc[Test_data.Outcome == 1, 'Outcome'] = 0


# train_data=pd.concat([train_data,weight])
# ÆŽœÓÁœžödataframe
# train_data.insert(1,'Pregrancies_weight',weight_1)
# train_data=train_data.replace(0,np.nan)

class Node():
    def __init__(self, data):
        self.lchild = None
        self.rchild = None
        self.data = data
        self.name = 0
        self.mean = 0
        # print(self.data['Outcome'].value_counts())
        k = np.array(self.data['Outcome'].value_counts())[0] / self.data.shape[0]
        Gini = 1 - (k * k) - (1 - k) * (1 - k)
        self.Gini = Gini


class Tree():
    def __init__(self):
        self.root = None
        self.features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                         'DiabetesPedigreeFunction', 'Age']
        self.len = len(self.features)

    def append(self, data):
        node = Node(data)

        if self.root == None:
            self.root = node
            return
        else:
            queue = [self.root]
            while queue:
                currunt = queue.pop(0)
                if currunt.lchild == None:
                    currunt.lchild = node
                    return node  # ÕâÀïÖ±œÓ·µ»ØŸÍ²»ÓÃÔÚŒÌÐøÑ­»·ÁË
                else:
                    queue.append(currunt.lchild)

                if currunt.rchild == None:
                    currunt.rchild = node
                    return node
                else:
                    queue.append(currunt.rchild)

    def creat_dataset(self, data, features_name):
        mini = data[features_name].min()
        max = data[features_name].max()
        mean = (mini + max) / 2
        data = data.sort_values(by=features_name)
        data_1 = data[(data.eval(features_name)) <= mean]
        data_2 = data[(data.eval(features_name)) > mean]
        D = data.shape[0]
        Dv_1 = data_1.shape[0]
        Dv_2 = data_2.shape[0]
        outcome_1 = np.array(data_1['Outcome'].value_counts())[0] / Dv_1
        outcome_2 = np.array(data_2['Outcome'].value_counts())[0] / Dv_2
        Gini_index = Dv_1 / (D) * (1 - outcome_1 ** 2 - (1 - outcome_1) ** 2) + Dv_2 / (D) * (
                    1 - outcome_2 ** 2 - (1 - outcome_2) ** 2)
        return data_1, data_2, Gini_index, mean

    def judge(self, node):
        Gini_index = []
        D = node.data.shape[0]
        for i in range(self.len):
            if len(node.data[self.features[i]].value_counts()) > 1:
                (data_1, data_2, Gini, mean) = self.creat_dataset(node.data, self.features[i])
                Gini_index.append(Gini)

        index = Gini_index.index(np.min(Gini_index))
        (data_1, data_2, m, mean) = self.creat_dataset(node.data, self.features[index])
        node.name = self.features[index]
        # print(node.name)
        node.mean = mean

        node.Gini = Gini_index[index]
        return index, data_1, data_2, mean, node

    def build(self, node):

        if self.root == None:
            self.root = node
        if node.Gini <= 0.10 or node.data.shape[0] <= 10:
            if node.Gini == 0:
                j = []
                for i in range(0, 2):
                    if i in node.data['Outcome'].value_counts():
                        j.append(i)
                if j[0] == 1:
                    node.name = 'true'
                    print('***')
                    return
                else:
                    node.name = 'false'
                    print('*')
                    return

            elif node.data['Outcome'].value_counts()[0] < 0.5:
                node.name = 'true'
                print('***')
                return

            else:

                node.name = 'false'
                return


        (k, data_1, data_2, m, node) = self.judge(node)
        node.name = self.features[k]
        node.mean = m
        # print(data_1.info())
        node_1 = self.append(data_1)
        node_2 = self.append(data_2)
        self.build(node_1)
        # print(data_2.info())

        self.build(node_2)

    def cut(self, data, node_1):
        print(node_1.name)
        index1 = data[(data.eval(node_1.name)) <= node_1.mean].index.tolist()
        index2 = data[(data.eval(node_1.name)) > node_1.mean].index.tolist()
        data_1 = data.loc[index1, :]
        data_2 = data.loc[index2, :]
        # print(data_1)
        # print(data_2)
        return data_1, data_2

    # ×îºÃ×öµœÖ±œÓÐÞžÄÊýŸÝµÄOutcome
    def apply(self, data, head):
        if head.name == 'true':
            print(1)
            data['Outcome'] = 1
            return
        if head.name == 'false':
            print(0)
            data['Outcome'] = 0
            return
        (data_1, data_2) = self.cut(data, head)

        lhead = head.lchild
        rhead = head.rchild
        # print(lhead.name)
        # print(rhead.name)

        '''if lhead.name=='true':
            data_1['Outcome']=1
            print(9)
            return
        if lhead.name=='false':
            data_1['Outcome']=0
            print(8)
            return
        if rhead.name=='true':
            data_2['Outcome']=1
            print(7)
            return 
        if rhead.name=='false':
            data_2['Outcome']=0
            print(7)
            return
        '''
        self.apply(data_1, lhead)
        self.apply(data_2, rhead)

tree = Tree()
tree.build(Node(Train_data))
tree.apply(Test_data, tree.root)
print(Test_data['Outcome'])
