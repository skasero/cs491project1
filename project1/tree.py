class Node(object):
    def __init__(self,key,leftHeuristic,rightHeuristic,ig):
        self.key = key
        self.leftHeuristic = leftHeuristic
        self.rightHeuristic = rightHeuristic
        self.ig = ig
        self.leaves = [] # left and right

    def add_node(self,node):
        self.leaves.append(node)