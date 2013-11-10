from data.stock import Stock
from predictor import Predictor
from score import auc

data, targets = Stock.train()
assert len(data) == len(targets)
print data[0]
print targets[0]
print len(data)
print len(targets)
print targets[:10]
half = len(data)/2
data, holdout_data = data[:half], data[half:]
targets, holdout_targets = targets[:half], targets[half:]
Predictor.train(data, targets)
preds = Predictor.predict(holdout_data)
print preds[0:50]

print auc(preds, holdout_targets)
