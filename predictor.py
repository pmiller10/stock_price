from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from models.stock_model import StockPredictor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

class Predictor:

    @classmethod
    def train(cls, data, targets):
        #model = LinearRegression()
        #model = SVR()
        #model = Ridge(alpha=0.001, tol=0.01)
        #model = ExtraTreesRegressor(n_estimators=5)
        #model = StockPredictor()
        #model = GradientBoostingRegressor(learning_rate=0.05, max_depth=3)
        model = RandomForestRegressor()
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
