from docclass import getwords
from classifier import classifier

# -- t9 --
class naivebayes(classifier.classifier):
    def __init__(self,getfeatures):
        classifier.__init__(self, getfeatures)
        self.thresholds={}

    def setthreshold(self,cat,t):
        self.thresholds[cat]=t

    def getthreshold(self,cat):
        if cat not in self.thresholds: return 1.0
        return self.thresholds[cat]

    def docprob(self, item, cat):
        features = self.getfeatures(item)
        # Multiply the probabilities of all the features together
        p = 1
        for f in features:
            p *= self.weightedprob(f, cat, self.fprob)
        return p

    def prob(self, item, cat):
        catprob = self.catcount(cat)/self.totalcount( )
        docprob = self.docprob(item, cat)
        return catprob * docprob

    # ---- t14 ----
    def classify(self,item,default=None):
        probs = {}
        # Find the category with the highest probability
        maximum = 0.0
        best = None
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > maximum:
                maximum = probs[cat]
                best = cat
        # Make sure the probability exceeds threshold*next best
        for cat in probs:
            if cat == best: continue
            if maximum < probs[cat] * self.getthreshold(best):
                return default
        return best


if __name__ == "__main__":
    cl = naivebayes(getwords)
    classifier.sampletrain(cl)
    print "'quick rabbit' is: " + str(cl.classify('quick rabbit', default='unknown'))
    print "'quick money' is: " + str(cl.classify('quick money', default='unknown'))
    cl.setthreshold('bad',1.0)
    print "'quick money' with 3 threshold in bad is: " + str(cl.classify('quick money', default='unknown'))
