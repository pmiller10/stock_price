from data.stock import Stock

data, targets = Stock.train(categorize=False)

pos, neg = [], []
for delta in targets:
    if delta > 0:
        pos.append(delta)
    elif delta < 0:
        neg.append(delta)

print sum(pos)/len(pos)
print sum(neg)/len(neg)
