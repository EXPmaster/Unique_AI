import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_file = './diabetes.csv'
dataset = pd.read_csv(data_file)
"""
sns.distplot(dataset['Pregnancies'])
plt.show()
sns.distplot(dataset['Glucose'])
plt.show()
sns.distplot(dataset['BloodPressure'])
plt.show()
sns.distplot(dataset['SkinThickness'])
plt.show()
sns.distplot(dataset['Insulin'])
plt.show()
sns.distplot(dataset['BMI'])
plt.show()
sns.distplot(dataset['DiabetesPedigreeFunction'])
plt.show()
sns.distplot(dataset['Age'])
# sns.relplot(x='Pregnancies', y='Age', hue='Outcome', data=dataset)
plt.show()
"""
length = dataset.shape[1]
label = dataset['Outcome']
dataset = dataset.drop(['Outcome'], axis=1)


