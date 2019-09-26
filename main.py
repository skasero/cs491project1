import numpy as np
import decision_trees as dt

if __name__ == "__main__":
	X = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
	Y = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])
	XV = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
	YV = np.array([[0], [0], [1], [0], [1], [1]])
	XT = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
	YT = np.array([[1], [1], [0], [0], [1], [0], [1], [1], [1]])

	max_depth = -1
	#DT = dt.DT_train_binary_best(X,Y,XV,YV)
	#print(DT)
	#DT = dt.DT_train_binary(X,Y,max_depth)
	#test_acc = dt.DT_test_binary(XT,YT,DT)
	#print(test_acc)

	dinnerX = np.array([[1, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1, 1]])
	dinnerY = np.array([[1], [1], [0], [1], [0], [1], [1], [0]])

	dinnerTX = np.array([[0, 1, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0]])
	dinnerTY = np.array([[0], [1], [0]])

	DT1 = dt.DT_train_binary(dinnerX[:5],dinnerY[:5],5)
	#print(dt.DT_test_binary(dinnerTX,dinnerTY,DT1))
	DT2 = dt.DT_train_binary(dinnerX[-5:],dinnerY[-5:],5)
	#print(dt.DT_test_binary(dinnerTX,dinnerTY,DT2))
	DT3 = dt.DT_train_binary(dinnerX[1:6],dinnerY[1:6],5)
	#print(dt.DT_test_binary(dinnerTX,dinnerTY,DT3))

	real = np.array([[4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 1.2], [5, 3.4, 1.6, 0.2], [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.7, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1]])
	realL = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0]])
	DT = dt.DT_train_real(real,realL,-1)

	exit(1)

























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

	#Traing set 1
	test8 = np.array([[0,1],[0,0],[1,0],[0,0],[1,1]])
	test8label = np.array([[1],[0],[0],[0],[1]])

	DT1 = DT_train_binary(test1,test1label,-1)
	print(DT_test_binary(test1,test1label,DT1))
	#DT2 = DT_train_binary(test4,test4label,5)
	#DT3 = DT_train_binary(test6,test6label,5)
	food = np.array([[0, 1, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0]])
	foodL = np.array([[0], [1], [0]])
	#print(DT_test_binary(food,foodL,DT1))
	#print(DT_test_binary(food,foodL,DT2))
	#print(DT_test_binary(food,foodL,DT3))
	#DT_train_binary(test6,test6label,5)
	#DT_train_binary(test8,test8label,-1)

	tx1 = np.array([[0, 1], [0, 0], [1, 0], [0, 0],  [1, 1]])
	ty1 = np.array([[1], [0], [0], [0], [1]])

	vx1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	vy1 = np.array([[0], [1], [0], [1]])

	tx2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
	ty2 = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])
	vx2 = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
	vy2 = np.array([[0], [0], [1], [0], [1], [1]])
	#DT = DT_train_binary(test1,test1label,-1)
	#accuracy = DT_test_binary(test1,test1label,DT)
	#print(DT_test_binary(test1,test1label,DT))
	#print(DT_train_binary_best(tx2,ty2,vx2,vy2))
	
	real = np.array([[4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 1.2], [5, 3.4, 1.6, 0.2], [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.7, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1]])
	realL = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0]])
	realVal = np.array([[4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 1.2]])
	realValL = np.array([[1],[1]])
	#dt_real = DT_train_real(real,realL,-1)
	#print(dt_real)
	#DT_test_real(real,realL,dt_real)
	#print(DT_test_real(real,realL,dt_real))
	#print(DT_train_real_best(real,realL,realVal,realValL))
