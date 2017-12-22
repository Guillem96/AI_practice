class Node:

    def __init__(self, state, parent=None, action=None, cost=0):
        self.parent = parent
        self.state = state
        self.cost = cost
        self.action = action

    def __str__(self):
        text =  "--Node {0} --\n" + \
                " Parent: {1}\n" + \
                " Action: {2}\n" + \
                " Cost: {3}"

        return text.format(self.state,
                            self.parent,
                            self.action,
                            self.cost)

    def __eq__(self, other):
         return self.state == other.state;

    #def r_path(self, node):
        #if node.parent == None:
            #return []

        #return [node.action] + self.r_path(node.parent)

    def path(self):
        sol = []
        while self.action != None:
            sol.append(self.action)
            self = self.parent
        sol.reverse()
        return sol
        # Recursive
        #res = self.r_path(self)
        #res.reverse()
        #return res

if __name__ == '__main__':
    n1 = Node("State1", None, "South")
    n2 = Node("State2", n1, "West")
    n3 = Node("State3", n2, "North")

    print n3.path()
