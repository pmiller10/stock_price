from data.stock import Stock
from predictor import Predictor
from score import auc
from preprocess import Preprocess

data, targets, cv_targets = Stock.train()

#data = Preprocess.scale(data)
#cv_targets = Preprocess.scale(cv_targets)
#data = Preprocess.polynomial(data, 5)
half = len(data)/2
tr_data, holdout_data = data[:half], data[half:]
train_targets, holdout_targets = targets[:half], targets[half:]

Predictor.train(tr_data, train_targets)
preds = Predictor.multi_predict(holdout_data)
print 'AUC ', auc(preds, cv_targets)
print Predictor.multi_predict([[0.]])
print Predictor.multi_predict([[0.1]])
print Predictor.multi_predict([[-0.1]])
print Predictor.multi_predict([[0.01]])
print Predictor.multi_predict([[-0.01]])
