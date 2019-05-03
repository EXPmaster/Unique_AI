import xgboost as xgb
from DataLoader import dataset, label
# read in data

trainset = dataset[:500]
trainlable = label[:500]
validset = dataset[500:]
validlable = label[500:]

dtrain = xgb.DMatrix(trainset, label=trainlable)
dtest = xgb.DMatrix(validset, label=validlable)
# specify parameters via map
param = {'max_depth': 1, 'eta': 1, 'objective': 'binary:logistic'}
num_round = 10
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)

label_pred = []
for i in range(len(preds)):
    if preds[i] > 0.5:
        label_pred.append(1)
    else:
        label_pred.append(0)

right = 0
for i in range(len(label_pred)):
    if validlable.iloc[i] == label_pred[i]:
        right += 1

print('accuracy: %lf' % (right / len(validlable)))
