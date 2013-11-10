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
    def train(cls):
        stocks = cls.load_train_data()
        stocks = [d[1:] for d in stocks] # remove ID
        opening_diffs = []
        closing_diffs = []
        open_vs_prior_close = []
        for stock in stocks:
            openings = [opening for opening in stock[0::5]]
            closings = [closing for closing in stock[3::5]]

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

        assert len(opening_diffs) == len(open_vs_prior_close)
        data = []
        for i,o in enumerate(opening_diffs):
            subdata = []
            subdata.append(o)
            subdata.append(open_vs_prior_close[i])
            data.append(subdata)

        #opening_diffs = [[item] for sublist in opening_diffs for item in sublist] # flatten
        #opening_diffs = [[d] for d in opening_diffs]
        probs = []
        for c in closing_diffs:
            if c > 0:
                probs.append(1)
            elif c < 0:
                probs.append(0)
            elif c == 0:
                probs.append(0.5)

        return data, probs

    @classmethod
    def test(cls):
        stocks = cls.load_test_data()
        opening_diffs = []
        headers = []
        for stock in stocks:
            diff = stock[-1] - stock[-6] 
            open_vs_close = stock[-1] - stock[-3]
            data = [diff, open_vs_close]
            opening_diffs.append(data)
            h = int((stock[0] * 100) + stock[1])
            headers.append(h)
        return opening_diffs, headers
