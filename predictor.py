from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LogisticRegression
from models.stock_model import StockPredictor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, ExtraTreesClassifier


class MyRegressor():
    
    def fit(self, data, targets):
        m = GradientBoostingRegressor(learning_rate=0.05)
        m.fit(data, targets)
        self.model = m

    def predict(self, data):
        return self.model.predict(data)[0]


class Predictor:

    @classmethod
    def train(cls, data, targets):
        cls.models = []
        #cls.models.append(MyRegressor)
        #cls.models.append(Ridge(alpha=1.1, tol=0.5))
        cls.models.append(LogisticRegression(penalty='l1', tol=0.01))
        #cls.models.append(ExtraTreesClassifier())
        #cls.models.append(LinearRegression())
        #model = SVR()
        #model = Ridge(alpha=0.001, tol=0.01)
        #model = ExtraTreesRegressor(n_estimators=5)
        #model = StockPredictor()
        #model = GradientBoostingRegressor(learning_rate=0.05, max_depth=3)
        #model = RandomForestRegressor()
        for m in cls.models:
            m.fit(data, targets)

    @classmethod
    def multi_predict(cls, data):
        preds = []
        for d in data:
            sub = []
            for m in cls.models:
                sub.append(m.predict(d)[0])
            p = sum(sub)/float(len(cls.models))
            preds.append(p)

        normed_preds = []
        for p in preds:
            if p > 1:
                normed_preds.append(1.)
            elif p < 0:
                normed_preds.append(0.)
            else:
                normed_preds.append(p)
        return normed_preds

    @classmethod
    def predict(cls, data):
        preds = [cls.model.predict(d) for d in data]
        normed_preds = []
        for p in preds:
            if p > 1:
                normed_preds.append(1.)
            elif p < 0:
                normed_preds.append(0.)
            else:
                normed_preds.append(p)
        return normed_preds

    @classmethod
    def stock_predict(cls, data):
        preds = cls.model.predict(data)
        normed_preds = []
        for p in preds:
            if p > 1:
                normed_preds.append(1.)
            elif p < 0:
                normed_preds.append(0.)
            else:
                normed_preds.append(p)
        return normed_preds


    @classmethod
    def votes(self):
        f1 = file('submissions/submission8.csv', 'r')
        f2 = file('submissions/submission5.csv', 'r')
        f3 = file('submissions/submission1.csv', 'r')
    
        votes1 = []
        votes2 = []
        votes3 = []
        for line in f1.readlines()[1:]:
            line = line.split(',')
            votes1.append(float(line[1]))
    
        for line in f2.readlines()[1:]:
            line = line.split(',')
            votes2.append(float(line[1]))
    
        for line in f3.readlines()[1:]:
            line = line.split(',')
            votes3.append(float(line[1]))

        final = []
        for i,v1 in enumerate(votes1):
            v2 = votes2[i]
            v3 = votes3[i]
            final.append((v1+v1+v2+v2+v3)/5.)

        return final
