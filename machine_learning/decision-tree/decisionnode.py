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

    def __str__(self):
        current_node = self
        result = ""

        if current_node.value != None:
            result += str(self.value) + " - " + str(self.col) + "\n"
        elif current_node.results:
            result += str(current_node.results) + "\n"

        if current_node.tb != None:
            result += "True: " + str(current_node.tb)

        if current_node.fb != None:
            result += "False: " + str(current_node.fb)

        return result
