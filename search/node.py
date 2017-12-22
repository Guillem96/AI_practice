class Node:

	def __init__(self,state,parent=None,action=None,cost=0):
		self.parent = parent
		self.state = state
		self.action = action
		self.cost = cost
	def __str__(self):
		return "--Node {0} --\n Parent: {1}\n Action: {2}\n Cost: {3}"\
		.format(self.state,self.parent,self.action,self.cost)
	def __eq__(self, other):
		return self.state == other.state
	def path(self):
		sol = []
		while self.action != None:
			sol.append(self.action)
			self = self.parent
		sol.reverse()
		return sol
if __name__ == "__main__":
	n1 = Node('state1')
	n2 = Node('state2',n1,'West',0)
	n3 = Node('state3',n2,'North',0)
	print n3
