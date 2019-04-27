import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# TODO one-hot

Train_file = './titanic/train.csv'
Test_file = './titanic/test.csv'
# Training set
trainset = pd.read_csv(Train_file)

trainset = trainset[['Pclass', 'Sex', 'Age',
                     'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
new_trainset = pd.DataFrame()

# Pclass
pclass_dummy = pd.get_dummies(
    trainset['Pclass'], prefix=trainset[['Pclass']].columns[0])
new_trainset = pd.concat([new_trainset, pclass_dummy], axis=1)

# Embarked
trainset['Embarked'] = trainset['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
trainset['Embarked'] = trainset['Embarked'].fillna(2)
embark_dummy = pd.get_dummies(
    trainset['Embarked'], prefix=trainset[['Embarked']].columns[0])
new_trainset = pd.concat([new_trainset, embark_dummy], axis=1)

# Sex
trainset['Sex'] = trainset['Sex'].map({'male': 0, 'female': 1})
sex_dummy = pd.get_dummies(trainset['Sex'],
                           prefix=trainset[['Sex']].columns[0])
new_trainset = pd.concat([new_trainset, sex_dummy], axis=1)

# trainset = trainset.fillna(method='ffill', axis=1)

# Age
new_trainset['Age'] = trainset['Age'].fillna(int(trainset['Age'].mean()))
new_trainset['Age'] = (new_trainset['Age'] - new_trainset['Age'].min()) / \
    (new_trainset['Age'].max() - new_trainset['Age'].min())

# Family Size
label1 = [0, 1, 2]
trainset['FamilySize'] = trainset['SibSp'] + trainset['Parch']
trainset['FamilySize'] = pd.cut(
    trainset['FamilySize'].T.values, 3, labels=label1)
family_dummy = pd.get_dummies(
    trainset['FamilySize'], prefix=trainset[['FamilySize']].columns[0])
new_trainset = pd.concat([new_trainset, family_dummy], axis=1)

# Fare
new_trainset['Fare'] = (trainset['Fare'] - trainset['Fare'].min()) / \
    (trainset['Fare'].max() - trainset['Fare'].min())
new_trainset['Survived'] = trainset['Survived']
# print(new_trainset)
label1 = [0, 1, 2]
label2 = [0, 1, 2, 3]

"""
trainset['Fare'] = pd.cut(trainset['Fare'].T.values, 3, labels=label1)
trainset['Age'] = pd.cut(trainset['Age'].T.values, 4, labels=label2)
"""

# sns.distplot(trainset['Fare'])
# sns.relplot(x='SibSp',y='Fare',hue='Survived',data=trainset)
# plt.show()
# Test set
testset = pd.read_csv(Test_file)
test_id = testset['PassengerId'].values
testset = testset[['Pclass', 'Sex', 'Age',
                   'SibSp', 'Parch', 'Fare', 'Embarked']]
testset['Embarked'] = testset['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
testset['Sex'] = testset['Sex'].map({'male': 0, 'female': 1})
testset['Age'] = testset['Age'].fillna(int(testset['Age'].mean()))
testset['Embarked'] = testset['Embarked'].fillna(2)
testset['SibSp'] = testset['SibSp'] + testset['Parch']
testset = testset.drop(['Parch'], axis=1)
testset.rename({'SibSp': 'FamilySize'}, inplace=True, axis='columns')

# testset['Fare'] = (testset['Fare'] - testset['Fare'].min()) / \
# (testset['Fare'].max() - testset['Fare'].min())
testset = testset.fillna(method='ffill', axis=1)
testset['Fare'] = pd.cut(testset['Fare'].T.values, 3, labels=label1)
testset['Age'] = pd.cut(testset['Age'].T.values, 4, labels=label2)

train_set = np.array(new_trainset, dtype=float)
test_data = np.array(testset, dtype=float)
print(train_set)
