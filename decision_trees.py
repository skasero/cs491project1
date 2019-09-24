#!/usr/bin/python3
import numpy as np
import tree as tree 

def DT_train_binary(X,Y,max_depth):
	h = calcHeuristic(X,Y) 
	DT = DT_train_binary_helper(X,Y,max_depth,h,0)
	print("root: " + str(DT.key))
	printTree(DT,1)
	return DT

def DT_train_binary_helper(X,Y,max_depth,h,currentDepth = 0,featuresUsed = list()):
	print("\n depth: ", currentDepth)
	# Found that there are no more Samples to check in X
	if(max_depth == -1 and len(X) <= 1):
		return None
	elif(max_depth != -1 and currentDepth >= max_depth):
		return None

	nodeArray = []
	for i in range(0,len(X[0])):
		if(i not in featuresUsed):
			leftHeuristic = calcHeuristic(X,Y,False,i)
			rightHeuristic = calcHeuristic(X,Y,True,i)
			ig = calcIG(h,leftHeuristic,rightHeuristic,X,i)
			n = tree.Node(i,leftHeuristic,rightHeuristic,ig)
			#print(str(leftHeuristic) + " " + str(rightHeuristic))
			print(ig)
			nodeArray.append(n)
	if(len(nodeArray) == 0):
		return None

	highestIG = 0
	index = 0
	featureNumber = 0
	for node in nodeArray:
		if(node.ig > highestIG):
			highestIG = node.ig
			featureNumber = node.key
			index = nodeArray.index(node)
	featuresUsed.append(featureNumber)

	node = nodeArray[index]
	if(node.leftHeuristic == 0 and node.rightHeuristic != 0):
		print("right")
		X,Y = removeFeature(0,X,Y,featureNumber)
		node.add_leaves(None,DT_train_binary_helper(X,Y,max_depth,node.rightHeuristic,currentDepth+1,featuresUsed))
	elif(node.leftHeuristic != 0 and node.rightHeuristic == 0):
		print("left")
		X,Y = removeFeature(1,X,Y,featureNumber)
		node.add_leaves(DT_train_binary_helper(X,Y,max_depth,node.leftHeuristic,currentDepth+1,featuresUsed),None)
	elif(node.leftHeuristic == 0 and node.rightHeuristic == 0):
		print("empty")
		node.add_leaves(None,None)
	elif(node.leftHeuristic != 0 and node.rightHeuristic != 0):
		print("2 splits")
		node.add_leaves(DT_train_binary_helper(X,Y,max_depth,node.leftHeuristic,currentDepth+1,featuresUsed),DT_train_binary_helper(X,Y,max_depth,node.rightHeuristic,currentDepth+1,featuresUsed))
	return node

def removeFeature(binary,X,Y,featureNumber):
	removeCount = 0
	for i in range(0,len(X)):
		if(X[i-removeCount][featureNumber] == binary):
			X = np.delete(X,i-removeCount,0)
			Y = np.delete(Y,i-removeCount,0)
			removeCount += 1
	return X,Y


# X = Feature array
# Y = Label array
# boolean = if the value in the feature we are checking is 0 or 1 (yes or no)
# index of the feature
# This funtion can also only take 2 paramaters. If 2 parameters are only called
# this function will calculate the heuristic based on the label array only.
# This featured would be used when trying to calc the heuristic of the init DT.
def calcHeuristic(X,Y,boolean = None,index = 0):
	search = 1 # either 0 or 1
	if(boolean is None):
		X = Y
	elif(boolean is False):
		search = 0

	count = 0
	total = 0
	for i in range(0,len(X)):
		# if feature is the same as search(yes/no)
		if(X[i][index] == search or boolean is None):
			total += 1
			# the label is 1(yes), aka counting the yeses
			if Y[i][0] == 1:
				count += 1
	
	if(total == 0):
		return 0
	percent = count/total # this is the percent of yeses
	# This checks if python will run into a RuntimeWarning b/c log(0)
	if(percent != 0 and (1-percent) != 0):
		h = percent*np.log2(percent)
		h += (1-percent)*np.log2(1-percent)
	else:
		h = 0
	return abs(h) # Absolute value

def calcIG(h,leftHeuristic,rightHeuristic,X,index):
	count = 0
	for item in X:
		if(item[index] == 1):
			count += 1
	percent = count/len(X)
	ig = h - ((1-percent)*leftHeuristic) - (percent*rightHeuristic)
	return ig

def printTree(node,depth):
	if(node is None):
		return
	print("Depth:" + str(depth) + " [", end='')
	for n in node.leaves:
		if n is None:
			print("Empty|", end='')
		else:
			print(str(n.key) + "|", end='')
	print("]")
	printTree(node.leaves[0],depth+1)
	printTree(node.leaves[1],depth+1)

def DT_test_binary(X,Y,DT):

	pass

if __name__ == "__main__":
	#first example in project
	X = np.array([[0,1,0,1],[1,1,1,1],[0,0,0,1]])
	Y = np.array([[1],[1],[0]])

	#netflix
	test1 = np.array([[1,1,0,0],[1,1,1,1],[1,1,1,1],[0,0,0,1],[0,0,1,1],[0,0,1,0],[0,0,0,0],[1,0,1,0],[1,1,1,0],[0,0,1,1]])
	test1label = np.array([[0],[1],[1],[0],[0],[1],[0],[0],[1],[0]])
	# all 8
	test2 = np.array([[1,1,1,0,1,1,0],[0,0,1,1,0,1,1],[0,1,0,0,1,0,0],[1,1,0,1,0,0,1],[1,0,1,0,1,1,1],[1,1,0,1,1,0,1],[1,1,0,0,1,1,0],[0,0,0,1,0,1,1]])
	test2label = np.array([[1],[1],[0],[1],[0],[1],[1],[0]])
	#first 5
	test3 = np.array([[1,1,1,0,1,1,0],[0,0,1,1,0,1,1],[0,1,0,0,1,0,0],[1,1,0,1,0,0,1],[1,0,1,0,1,1,1]])
	test3label = np.array([[1],[1],[0],[1],[0]])
	#last 5
	test4 = np.array([[1,1,0,1,0,0,1],[1,0,1,0,1,1,1],[1,1,0,1,1,0,1],[1,1,0,0,1,1,0],[0,0,0,1,0,1,1]])
	test4label = np.array([[1],[0],[1],[1],[0]])
	#middle 2-6
	test5 = np.array([[0,0,1,1,0,1,1],[0,1,0,0,1,0,0],[1,1,0,1,0,0,1],[1,0,1,0,1,1,1],[1,1,0,1,1,0,1]])
	test5label = np.array([[1],[0],[1],[0],[1]])
	#middle 3-7
	test6 = np.array([[0,1,0,0,1,0,0],[1,1,0,1,0,0,1],[1,0,1,0,1,1,1],[1,1,0,1,1,0,1],[1,1,0,0,1,1,0]])
	test6label = np.array([[0],[1],[0],[1],[1]])
	max_depth = -1
	#custom
	test7 = np.array([[1,1,0],[1,1,0],[1,1,1],[1,0,0],[1,0,1],[0,0,1],[0,1,1],[0,1,0],[0,0,0]])
	test7label = np.array([[1],[1],[1],[0],[0],[1],[1],[0],[0]])

	#Traing set 1
	test8 = np.array([[0,1],[0,0],[1,0],[0,0],[1,1]])
	test8label = np.array([[1],[0],[0],[0],[1]])

	DT_train_binary(test1,test1label,max_depth)
	#DT_train_binary(test3,test3label,5)
	#DT_train_binary(test4,test4label,5)
	#DT_train_binary(test5,test5label,5)
	#DT_train_binary(test6,test6label,5)
	#DT_train_binary(test7,test7label,-1)
	#DT_train_binary(test8,test8label,-1)



