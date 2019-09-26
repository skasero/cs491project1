#!/usr/bin/python3

# Project 1
# Description: Decision Trees
# Authors: Spencer Kase Rohlfing & Dhruv Bhatt 
# 9/26/19

import numpy as np
import sys # Used for getting a max value
import math # Used for log, sorry np.log2 is trash because I can't use try/except blocks

#################################################################
# # # # # # # # # # # # # DT_TRAIN_BINARY # # # # # # # # # # # #
#################################################################

def DT_train_binary(X,Y,max_depth):
	h = calcEntropy(Y) 
	# print(h)
	featuresUsed = []
	for i in range(0,len(X[0])):
		featuresUsed.append(i)
	DT = DT_train_binary_helper(X,Y,max_depth,h,0,featuresUsed)
	# print(DT)
	return DT

def DT_train_binary_helper(X,Y,max_depth,h,currentDepth,featuresUsed):
	# Checking if were at the current max depth or if all features have been used
	# Also checks if h is 0
	if(max_depth == currentDepth or not featuresUsed or h==0 and currentDepth != 0):
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

		hNo = calcEntropy(yNo)
		hYes = calcEntropy(yYes)
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
# this function will calculate the Entropy based on the label array only.
# This featured would be used when trying to calc the Entropy of the init DT.
def calcEntropy(Y):
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

# Used to calculate the Information Gain of a split
def calcIG(h,leftEntropy,rightEntropy,X,index):
	count = 0
	for item in X:
		if(item[index] == 1):
			count += 1
	percent = count/len(X)
	ig = h - ((1-percent)*leftEntropy) - (percent*rightEntropy)
	return ig

# Used to find if there are more 0s or 1s in a feature
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

		
##################################################################
# # # # # # # # # # # # # DT_TEST_BINARY # # # # # # # # # # # # #
##################################################################
def DT_test_binary(X,Y,DT):
	correntCount = 0
	for i in range(0,len(X)):
		result = DT_make_prediction(X[i],DT)
		# print(result)
		if(result == Y[i]):
			correntCount +=1

	# print(correntCount / len(X))
	return correntCount / len(X)

################################################################
# # # # # # # # # # # DT_MAKE_PREDICTION # # # # # # # # # # # # 
################################################################
def DT_make_prediction(x,DT):
	if(isinstance(DT,int)):
		return DT
	# Left Side
	if(x[DT[0]] == 0):
		return DT_make_prediction(x,DT[1])
	# Right side
	elif(x[DT[0]] == 1):
	 	return DT_make_prediction(x,DT[2])

################################################################
# # # # # # # # # # # DT_TRAIN_BINARY_BEST # # # # # # # # # # # 
################################################################
def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
	dtArray = []
	for i in range(0,len(X_train[0])):
		dtArray.append(DT_train_binary(X_train,Y_train,i+1))

	bestDT = None
	bestAccuracy = 0
	for dt in dtArray:
		accuracy = DT_test_binary(X_val,Y_val,dt)
		if(accuracy > bestAccuracy):
			bestAccuracy = accuracy
			bestDT = dt
	# print(bestAccuracy)
	return bestDT

###############################################################
# # # # # # # # # # # # # DT_TRAIN_REAL # # # # # # # # # # # # 
###############################################################
def DT_train_real(X,Y,max_depth):
	avgs = getAverages(X)
	X_converted = convertToBinary(X,avgs)
	DT = DT_train_binary(X_converted,Y,max_depth)
	DT.append(avgs)
	return DT

##############################################################
# # # # # # # # # # # # # DT_TEST_REAL # # # # # # # # # # # # 
##############################################################
def DT_test_real(X,Y,DT):
	avgs = DT[len(DT)-1]
	X_converted = convertToBinary(X,avgs)
	DT_converted = DT[:-1]
	accuracy = DT_test_binary(X_converted,Y,DT_converted)
	return accuracy
################################################################
# # # # # # # # # # # DT_TRAIN_REAL_BEST # # # # # # # # # # # 
################################################################
def DT_train_real_best(X_train,Y_train,X_val,Y_val):
	avgs = getAverages(X_train)
	X_train_converted = convertToBinary(X_train,avgs)
	X_val_converted = convertToBinary(X_val,avgs)
	DT = DT_train_binary_best(X_train_converted,Y_train,X_val_converted,Y_val)
	return DT

# Returns a list of averages for the features
def getAverages(X):
	# lol this is the moment I found out about list comprehension 
	avgs = [sum(i)/len(X) for i in zip(*X)]
	# print(avgs)
	output = convertToBinary(X,avgs)
	return avgs

# Takes in samples and averages list and returns a sample list
# based on if the real number was less than or greater than the average
# value for each feature
def convertToBinary(X,avgs):
	output = []
	for item in X:
		tmp = []
		for i in range(0,len(item)):
			if(item[i] > avgs[i]):
				tmp.append(1)
			else:
				tmp.append(0)
		output.append(tmp)
	return output
