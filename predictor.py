from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

class Predictor:

    @classmethod
    def train(cls, data, targets):
        #model = LinearRegression()
        #model = SVR()
        model = Ridge(alpha=0.9)
        model.fit(data, targets)
        cls.model = model

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
