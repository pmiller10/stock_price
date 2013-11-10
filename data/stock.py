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
        all_openings = []
        for stock in stocks:
            openings = []
            for opening in stock[0::5]:
                openings.append(opening)
            diffs = []
            for i,o in enumerate(openings[1:]):
                diff = o - openings[i-1]
                diffs.append(diff)
            opening_diffs.append(diffs)
            all_openings += openings[1:]

            closings = []
            for closing in stock[3::5]:
                closings.append(closing)
            diffs = []
            for i,c in enumerate(closings[1:]):
                diff = c - openings[i+1]
                diffs.append(diff)

            closing_diffs.append(diffs)

        opening_diffs = [[item] for sublist in opening_diffs for item in sublist] # flatten
        closing_diffs = [item for sublist in closing_diffs for item in sublist] # flatten
        probs = []
        for c in closing_diffs:
            if c > 0:
                probs.append(1)
            elif c < 0:
                probs.append(0)
            elif c == 0:
                #print 'is 0'
                probs.append(0.5)

        return opening_diffs, probs

    @classmethod
    def test(cls):
        stocks = cls.load_test_data()
        opening_diffs = []
        headers = []
        for stock in stocks:
            diff = stock[-1] - stock[-6] 
            opening_diffs.append([diff])
            h = int((stock[0] * 100) + stock[1])
            headers.append(h)
        return opening_diffs, headers
