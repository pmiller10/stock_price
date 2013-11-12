from preprocess import Preprocess
from sklearn import preprocessing
import math

class Stock:

    @classmethod
    def load_test_data(cls):
        f = file('data/test.csv', 'r')
        return cls.load_data(f)

    @classmethod
    def load_train_data(cls):
        f = file('data/training.csv', 'r')
        return cls.load_data(f)

    @classmethod
    def load_data(cls, f):
        stocks = []
        for line in f.readlines()[1:]: # remove header
            line = line.split(',')
            line = [float(l) for l in line]
            stocks.append(line)
        return stocks

    @classmethod
    def train(cls, categorize=True):
        stocks = cls.load_train_data()
        stocks = [d[1:] for d in stocks] # remove ID
        opening_diffs = []
        closing_diffs = []
        open_vs_prior_close = []
        volumes = []
        max_vs_open = []
        for stock in stocks:
            openings = [opening for opening in stock[0::5]]
            closings = [closing for closing in stock[3::5]]
            volumes += [v for v in stock[9::5]] # start at 9 to avoid day 1
            maxs = [v for v in stock[1::5]]

            # opening diffs
            for i,o in enumerate(openings[1:]):
                diff = o - openings[i-1]
                opening_diffs.append(diff)

            # closing diffs
            for i,c in enumerate(closings[1:]):
                diff = c - openings[i+1]
                closing_diffs.append(diff)

            # opening vs prior day close diff
            for i,o in enumerate(openings[1:]):
                diff = o - closings[i]
                open_vs_prior_close.append(diff)

            # max vs opening
            for i,o in enumerate(openings[1:]):
                diff = o - maxs[i]
                max_vs_open.append(diff)

        assert len(opening_diffs) == len(open_vs_prior_close) == len(volumes) == len(max_vs_open)

        print 'before ', closing_diffs[:20] 
        print 'avg ', sum(closing_diffs)/float(len(closing_diffs))
        directions = cls.norm5(closing_diffs)
        print 'avg ', sum(directions)/float(len(directions))
        print 'max ', max(directions)
        print 'min ', min(directions)
        print 'after ', directions[:20] 

        data = []
        for i,o in enumerate(opening_diffs):
            subdata = []
            subdata.append(o)
            #subdata.append(open_vs_prior_close[i]) # don't use
            #subdata.append(volumes[i])
            #subdata.append(max_vs_open[i])
            data.append(subdata)

        half = len(closing_diffs)/2
        return data, directions, cls.norm2(closing_diffs[half:])

    @classmethod
    def test(cls):
        stocks = cls.load_test_data()
        opening_diffs = []
        headers = []
        for stock in stocks:
            diff = stock[-1] - stock[-6] 
            open_vs_close = stock[-1] - stock[-3]
            open_vs_max = stock[-1] - stock[-5]

            data = []
            data.append(diff)
            #data.append(open_vs_max) # don't use
            #data.append(open_vs_close) # don't use
            opening_diffs.append(data)
            h = int((stock[0] * 100) + stock[1])
            headers.append(h)
        return opening_diffs, headers

    @classmethod
    def norm(cls, data):
        small = min(data) * -1.
        positives = [i+small for i in data]
        big = max(positives)
        normed = [i/big for i in positives]
        return normed 

    @classmethod
    def norm2(cls, data):
        norms = []
        for c in data:
            if c > 0:
                norms.append(1)
            elif c < 0:
                norms.append(0)
            elif c == 0:
                norms.append(0.5)
        return norms

    @classmethod
    def norm3(cls, data):
        probs = []
        pos = 0.13
        neg = -0.13
        for c in data:
            if c >= pos:
                probs.append(1)
            elif c < pos and c > 0:
                probs.append(0.75)
            elif c <= neg:
                probs.append(0)
            elif c < 0 and c > neg:
                probs.append(0.25)
            elif c == 0:
                probs.append(0.5)
            else:
                print Warning('didnt fit: '.format(c))
        return probs

    @classmethod
    def norm4(cls, data):
        data = [[d] for d in data]
        print 'deviation ', Preprocess.standard_deviation(data)
        data = [d[0] for d in data]

        data = Preprocess.root(data, 2)
        data = Preprocess.squeeze(data)
        data = Preprocess.squeeze(data)

        #data = [[d] for d in data]
        #data = Preprocess.scale(data)
        #data = [d[0] for d in data]

        data = [[d] for d in data]
        print 'deviation ', Preprocess.standard_deviation(data)
        #data = preprocessing.normalize(data, norm='l2')
        data = Preprocess.norm(data)
        print 'deviation ', Preprocess.standard_deviation(data)
        data = [d[0]-0.04 for d in data]
        data = [round(d, 1) for d in data]

        return data

    @classmethod
    def norm5(cls, data):
        data = [[d] for d in data]
        print 'deviation ', Preprocess.standard_deviation(data)
        #data = Preprocess.scale(data)
        #print 'deviation ', Preprocess.standard_deviation(data)
        data = [(d[0]*10)+0.5 for d in data]

        return data
