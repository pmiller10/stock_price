class Stock:

    @classmethod
    def load_data(cls):
        f = file('training.csv', 'r')
        stocks = []
        for line in f.readlines()[1:]: # remove header
            line = line.split(',')
            line = [float(l) for l in line]
            stocks.append(line)
        return stocks

    @classmethod
    def training(cls):
        stocks = cls.load_data()
        stocks = [d[1:] for d in stocks] # remove ID
        opening_diffs = []
        for stock in stocks:
            openings = []
            for opening in stock[0::5]:
                openings.append(opening)
            diffs = []
            for i,o in enumerate(openings[1:]):
                diff = o - openings[i-1]
                diffs.append(diff)
            opening_diffs.append(diffs)
        return opening_diffs
