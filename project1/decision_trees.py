#!/usr/bin/python3
import numpy as np
import tree as tree 

def DT_train_binary(X,Y,max_depth):
    h = calcHeuristic(Y,0)
    pass

# Y = Label array
# index is the index of the 
def calcHeuristic(Y,boolean,index = None,X = None):
    count = {}
    for item in X:
        if not item[index] in count.keys():
            count[item[index]] = 1
        else:
            count[item[index]] += 1
    
    total = 0
    for item in count:
        total += count.get(item)

    h = 0
    for item in count:
        percent = count.get(item) / total
        h += percent*np.log2(percent)

    return abs(h) # Absolute value

if __name__ == "__main__":
    X = np.array([[0,1,0,1],[1,1,1,1],[0,0,0,1]])
    Y = np.array([[1],[1],[0]])
    max_depth = -1
    DT_train_binary(X,Y,max_depth)