#!/usr/bin/env python

class decisionnode:

    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None, parent=None, answer=True):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb # True results
        self.fb = fb # False results
        self.parent = parent
        self.answer = answer

    def is_leaf(self):
        return self.results != None
