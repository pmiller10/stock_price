from data.stock import Stock
from predictor import Predictor
from score import auc

def submission(ids, preds):
    name = 'submissions/submission1.csv'
    f = open(name, 'w')
    if len(ids) != len(preds):
        raise Exception("The number of IDs and the number of predictions are different")
    string = 'id,prediction\n'
    for index,i in enumerate(ids):
        string += str(i) + ',' + str(preds[index]) + "\n"
    f.write(str(string))
    f.close()

data, targets = Stock.train()
holdout_data, ids = Stock.test()
assert len(data) == len(targets)
print len(holdout_data)
print len(ids)

Predictor.train(data, targets)
preds = Predictor.predict(holdout_data)
print preds[0:50]
submission(ids, preds)
