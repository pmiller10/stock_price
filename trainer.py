from data.stock import Stock
from predictor import Predictor
from score import auc
from preprocess import Preprocess

data, train_probs, cv_targets = Stock.train()
assert len(data) == len(train_probs)
assert len(data) == len(cv_targets)


half = len(data)/2
#data = Preprocess.polynomial(data, 2)
data, holdout_data = data[:half], data[half:]
#targets, holdout_targets = train_probs[:half], cv_targets[half:]
#print targets[:50]

################################
#spc_train_data= []
#spc_hold_data= []
#spc_train_targets= []
#spc_hold_targets= []
#
#count = 0
#for i,d in enumerate(data):
#    count += 1
#    t = cv_targets[i]
#    if count <= 250:
#        spc_train_data.append(d)
#        spc_train_targets.append(t)
#    elif count > 250 and count <= 499:
#        spc_hold_data.append(d)
#        spc_hold_targets.append(t)
#    else: print Warning('count ', count)
#
#    if count >= 499: count = 0
#
#print len(spc_train_data)
#print len(spc_hold_data)
#print len(spc_train_targets)
#print len(spc_hold_targets)
#
#Predictor.train(spc_train_data, spc_train_targets)
#preds = Predictor.predict(spc_hold_data)
#print preds[:50]
#print auc(preds, spc_hold_targets)
################################

Predictor.train(data, cv_targets)
preds = Predictor.predict(data)
#print preds[0:50]

print auc(preds, holdout_targets)
