from docclass import getwords
# ---- t2 ----
class classifier:
    def __init__(self, getfeatures, filename=None):
        # Counts of feature/category combinations
        self.fc = {}
        # Counts of documents in each category
        self.cc = {}
        self.getfeatures = getfeatures
        self.filename = filename

    # Increase the count of a feature/category pair
    # ---- t3 ----
    def incf(self,f,cat):
        self.fc.setdefault(f,{})
        self.fc[f].setdefault(cat,0)
        self.fc[f][cat] += 1

    # Increase the count of a category
    def incc(self,cat):
        self.cc.setdefault(cat,0)
        self.cc[cat] += 1

    # The number of times a feature is in a category
    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])

        return 0.0

    # The number of items in a category
    def catcount(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0.0

    # The total number of items
    def totalcount(self):
        total = 0
        for cat in self.cc.keys():
            total += self.cc[cat]

        return total

    # The list of all categories
    def categories(self):
        return self.cc.keys()

    # ---- t4 ----
    def train(self,item,cat):
        features = self.getfeatures(item)
        # Increment the count for every
        # feature with this category
        for f in features:
            self.incf(f, cat)
        # Increment the count for this category
        self.incc(cat)

    # ---- t6 ----
    def fprob(self, f, cat):
        if self.catcount(cat) == 0: return 0
        # The total number of times this feature appeared
        # in this category divided by the total number of
        # items in this category
        return self.fcount(f,cat)/self.catcount(cat)

    def weightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
        # Calculate current probability
        basicprob = prf(f, cat)
        # Count the number of times this feature has appeared in
        # all categories
        totals = sum(map(lambda c: self.fcount(f, c), self.fc[f].keys()))
        # Calculate the weighted average
        wp = (weight * ap + totals * basicprob)/(totals+weight)
        return wp

    # ---- t5 ----
    @staticmethod
    def sampletrain(cl):
        cl.train('Nobody owns the water.','good')
        cl.train('the quick rabbit jumps fences','good')
        cl.train('buy pharmaceuticals now','bad')
        cl.train('make quick money at the online casino','bad')
        cl.train('the quick brown fox jumps','good')


if __name__ == '__main__':
    c = classifier(getwords)
    classifier.sampletrain(c)
    print "---- t7 ----\nWeighted probability of money with one sample train: " + \
            str(c.weightedprob('money', 'good', c.fprob))
    classifier.sampletrain(c)
    print "Weighted probability of money with two sample trains: " + \
            str(c.weightedprob('money', 'good', c.fprob))
