#!/usr/bin/python3
import numpy as np
import sys # Used for getting a max value
import math # Used for log, sorry np.log2 is trash because I can't use try/except blocks

def DT_train_binary(X,Y,max_depth):
	h = calcHeuristic(Y) 
	# print(h)
	featuresUsed = []
	for i in range(0,len(X[0])):
		featuresUsed.append(i)
	DT = DT_train_binary_helper(X,Y,max_depth,h,0,featuresUsed)
	print(DT)
	return DT

def DT_train_binary_helper(X,Y,max_depth,h,currentDepth,featuresUsed):
	# Checking if were at the current max depth or if all features have been used
	# Also checks if h is 0
	if(max_depth == currentDepth or not featuresUsed or h==0):
		return findGreatestLabel(Y)
	
	# This is like -INT_MAX in C++. I could just use a small number, but...
	# These variables are used to keep track of the best values based on which feature
	# has the highest information gain (IG)
	bestIG = -sys.maxsize-1 
	bestFeature = -sys.maxsize-1 
	bestHNo = -sys.maxsize-1 
	bestHYes = -sys.maxsize-1 
	bestXNo = []
	bestXYes = []
	bestYNo = []
	bestYYes = []

	# Loops through all possible features still in use
	for item in featuresUsed:
		xNo = []
		xYes = []
		yNo = []
		yYes = []
		for i in range(0,len(X)):
			if(X[i][item] == 0):
				xNo.append(X[i])
				yNo.append(Y[i])
			else:
				xYes.append(X[i])
				yYes.append(Y[i])

		hNo = calcHeuristic(yNo)
		hYes = calcHeuristic(yYes)
		# print(hNo)
		# print(hYes)

		ig = calcIG(h,hNo,hYes,X,item)
		# print(ig)

		# If found a better IG, then store all the variables from feature
		if(ig > bestIG):
			bestIG = ig
			bestFeature = item
			bestHNo = hNo
			bestHYes = hYes
			bestXNo = xNo
			bestYNo = yNo
			bestXYes = xYes
			bestYYes = yYes

	# Remove best feature choosen
	featuresUsed.remove(bestFeature)
	# print(featuresUsed)

	# Increment depth by 1
	currentDepth +=1
	# print(currentDepth)

	return([bestFeature,DT_train_binary_helper(bestXNo,bestYNo,max_depth,bestHNo,currentDepth,featuresUsed),DT_train_binary_helper(bestXYes,bestYYes,max_depth,bestHYes,currentDepth,featuresUsed)])

# X = Feature array
# Y = Label array
# boolean = if the value in the feature we are checking is 0 or 1 (yes or no)
# index of the feature
# This funtion can also only take 2 paramaters. If 2 parameters are only called
# this function will calculate the heuristic based on the label array only.
# This featured would be used when trying to calc the heuristic of the init DT.
def calcHeuristic(Y):
	yesCount = 0
	total = 0
	for item in Y:
		if item == 1:
			yesCount +=1
		total +=1 


	# This checks if python will run into a RuntimeWarning b/c log(0)
	try:
		percent = yesCount/total # this is the percent of yeses
		h = percent*math.log2(percent)
		h += (1-percent)*math.log2(1-percent)
	except:
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

def findGreatestLabel(Y):
	count = {
		0:0,
		1:0
	}
	for item in Y:
		count[item[0]] += 1
	if(count[0] >= count[1]):
		return 0
	else:
		return 1

def DT_test_binary(X,Y,DT):
	correntCount = 0
	for i in range(0,len(X)):
		result = DT_test_binary_helper(X[i],DT)
		if(result == Y[i]):
			correntCount +=1

	# print(correntCount / len(X))
	return correntCount / len(X)

def DT_test_binary_helper(X,DT):
	# Left Side
	if(isinstance(DT[1],int) and X[DT[0]] == 0):
		return DT[1]
	# Right side
	elif(isinstance(DT[2],int) and X[DT[0]] == 1):
		return DT[2]
	else:
		if(X[DT[0]] == 0):
			return DT_test_binary_helper(X,DT[1])
		else:
			return DT_test_binary_helper(X,DT[2])

if __name__ == "__main__":
	#first example in project
	X = np.array([[0,1,0,1],[1,1,1,1],[0,0,0,1]])
	Y = np.array([[1],[1],[0]])

	#netflix
	test1 = np.array([[1,1,1,0],[1,1,1,1],[1,1,1,1],[0,0,0,1],[0,0,1,1],[0,0,1,0],[0,0,0,0],[1,0,1,0],[1,1,1,0],[0,0,1,1]])
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

	DT = DT_train_binary(test1,test1label,-1)
	#DT_train_binary(test3,test3label,5)
	#DT_train_binary(test4,test4label,5)
	#DT_train_binary(test5,test5label,5)
	#DT_train_binary(test6,test6label,5)
	#DT_train_binary(test7,test7label,-1)
	#DT_train_binary(test8,test8label,-1)
	accuracy = DT_test_binary(test1,test1label,DT)