class bicluster:
    def __init__(self,vec,left=None,right=None,dist=0.0,id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = dist
