from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

class StockPredictor:

    def fit(self, data, targets):
        size = 499
        data_groups = self.group_data(data, size)
        target_groups = self.group_data(targets, size)
        models = []
        for i,data in enumerate(data_groups):
            targets = target_groups[i]
            #m = LinearRegression()
            m = Ridge(alpha=0.001, tol=0.0001)
            m.fit(data, targets)
            models.append(m)
        self.models = models

    def group_data(self, data, size):
        print '...incoming data ', len(data)
        print '...group size ', size
        assert len(data) % size == 0
        count = 0
        groups = []
        group = [] # first group
        for i,d in enumerate(data):
            count += 1
            group.append(d)
            if count == size:
                count = 0
                groups.append(group)
                group = []
        return groups

    def predict(self, data):
        count = 0
        preds = []
        groups = []
        for i,d in enumerate(data):
            m = i % 94
            pred = self.models[m].predict(d)
            preds.append(pred)
        return preds
