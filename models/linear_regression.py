from sklearn import metrics

class AucLinearRegression():

    def __init__(self):
        self.bias = 0.5
        self.weight = 1.1
        self.learning_rate = 0.0003
        self.iterations = 10

    def fit(self, data, targets):
        preds = self.predict_all(data, self.weight)
        cost = self.cost(preds, targets)
        sign = True # sign of the gradient
        for i in range(self.iterations):
            self.learning_rate = 1/float(i+1.)#/float(self.iterations))
            #print 'learning rate ', self.learning_rate
            weight, cost, sign = self.derivative(data, targets, cost, sign)
            self.weight = weight
            print self.weight, 1-cost, sign
        self.weight = -0.5

    def derivative(self, data, targets, old_cost, positive=True):
        weight = self.weight+self.learning_rate if positive else self.weight-self.learning_rate
        preds = self.predict_all(data, weight)
        cost = self.cost(preds, targets)
        #print 'new cost is ', cost, ' compared to old cost ', old_cost
        better = cost <= old_cost
        if positive and better:
            sign = True
        elif not positive and better:
            sign = False
        elif positive and not better:
            sign = False
        elif not positive and not better:
            sign = True
            
        if better:
            return weight, cost, sign
        else:
            return self.weight, old_cost, sign

    def predict_all(self, data, weight):
        return [self.predict(d, weight) for d in data]

    def predict(self, data, weight=None):
        if not weight: weight = self.weight
        return self.bias + (weight * data[0])

    def cost(self, preds, targets):
        fpr, tpr, thresholds = metrics.roc_curve(targets, preds, pos_label=1)
        auc = metrics.auc(fpr,tpr)
        return 1 - auc
