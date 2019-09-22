class Node(object):

    # key is index
    # leftHeuristic is left branches heuristic value
    def __init__(self,key,leftHeuristic,rightHeuristic,ig):
        self.key = key 
        self.leftHeuristic = leftHeuristic
        self.rightHeuristic = rightHeuristic
        self.ig = ig
        self.leaves = [] # left and right

    def add_leaves(self,leftNode,rightNode):
        self.leaves.append(leftNode)
        self.leaves.append(rightNode)