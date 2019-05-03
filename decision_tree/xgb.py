import xgboost as xgb
# read in data
data_file = './diabetes.csv'
dtrain = xgb.DMatrix(data_file)
dtest = xgb.DMatrix(data_file)
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)