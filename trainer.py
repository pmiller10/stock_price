from data.stock import Stock
from predictor import Predictor
from score import auc
from preprocess import Preprocess

data, targets, cv_targets = Stock.train()

#data = Preprocess.scale(data)
#cv_targets = Preprocess.scale(cv_targets)
half = len(data)/2
tr_data, holdout_data = data[:half], data[half:]
train_targets, holdout_targets = targets[:half], targets[half:]

Predictor.train(tr_data, train_targets)
preds = Predictor.multi_predict(holdout_data)
print 'preds ', preds[:20]
#stop = 10
#for t in holdout_targets[:stop]: print t
#print preds[:stop]
#print holdout_targets[:stop]
#print auc(preds[:stop], holdout_targets[:stop])
print auc(preds, cv_targets)
