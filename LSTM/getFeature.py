import os

import numpy as np
import matplotlib.pyplot as plt
import math


from getData import Trainset



def trafficData():
	#获取traffic的文件列表，文件名列表
	file_list = os.listdir('traffic-fix/raw')
	file_name = []
	for file in file_list:
		file_name.append(file.split('.')[0])

	file_path = os.path.join('traffic-fix/traffic', file_list[0])
	with open(file_path,'r') as file:
    	#将所有行读取
		ids = []
		labels = []
		current_t = []
		time_diff = []
		prediction_t = []

		lines = file.readlines()
		
		for line in lines:
			splitBySpace = line.split(' ')
			ids.append(splitBySpace[0])
			labels.append(splitBySpace[1])
			current_t.append(splitBySpace[2])
			prediction_t.append(splitBySpace[3].split(';')[0])

			time_diff.append(int(splitBySpace[3].split(';')[0]) - int(splitBySpace[2])) 

	return max(time_diff)

def getRspeed(speed, index):

	file_name = str(index) + '.txt'

	with open(os.path.join('attr',file_name),'r') as f:

		lines = f.readlines()
	
	line = lines[0].split('	')

	vk = float(line[6])

	

	return speed/vk

def getLevel(index):

	with open('attr/'+str(index)+'.txt','r') as f:

		lines = f.readlines()

		line = lines[0].split('\t')

		level = int(line[1])

	return level

def stateVspeed():

	trainset = Trainset('test/27.777778')
	labels = []
	speeds = []
	ep = 0.001

	for i,data in enumerate(trainset,0):
		
		if i + 1 < 0:
			continue

		if i + 1 == 2037:
			break

		if data[2][0][2] != 3:
			continue

		labels.append(data[5] - 4)

		speeds.append(data[2][0][0])

		

	plt.scatter(speeds, labels)
	plt.show()

def speedDvL():

	trainset = Trainset()
	diffs = []
	lns = []
	ep = 0.01

	for data in trainset:

		if len(lns) > 1000:
			break

		if data[2][0][2] != 4:
			continue

		index = data[4]
		# for seq in data[2]:
		# 	labels.append(seq[2])
		# 	density = getDensity(seq[3], index)
		# 	densitys.append(density)

		

		lns.append(data[2][0][0])

		rspeed = getRspeed(data[2][0][0], index)
		diffs.append(rspeed)



	plt.scatter(lns, diffs)
	plt.show()

def getMDn():

	

	dataset = Trainset('20190724')

	num = len(dataset)

	print(num)

	M = 0.0

	for i,data in enumerate(dataset,0):

		M = data[2][0][3] + data[2][1][3] + data[2][2][3] + data[2][3][3] + data[2][4][3] + M

		if (i + 1) == 2000:

			break

	M = M / (2000 * 5)

	print(M) 

	D = 0.0

	for i,data in enumerate(dataset,0):

		D = pow((data[2][0][3] - M),2) + pow((data[2][1][3] - M),2) + pow((data[2][2][3] - M),2) + pow((data[2][3][3] - M),2) + pow((data[2][4][3] - M),2) + D

		if (i + 1) == 2000:

			break

	D = D / (2000 * 5)

	D = math.sqrt(D)

	print(D) 

	return num, M, D

def getMD(vk):

	root = os.path.join('test',str(vk))

	dataset = Trainset(root)

	num = len(dataset)

	print(num)

	M = 0.0

	for i,data in enumerate(dataset,0):

		M = data[2][0][0] + data[2][1][0] + data[2][2][0] + data[2][3][0] + data[2][4][0] + M

		if (i + 1) == len(dataset):

			break

	M = M / (num * 5)

	print(M) 

	D = 0.0

	for i,data in enumerate(dataset,0):

		D = pow((data[2][0][0] - M),2) + pow((data[2][1][0] - M),2) + pow((data[2][2][0] - M),2) + pow((data[2][3][0] - M),2) + pow((data[2][4][0] - M),2) + D

		if (i + 1) == len(dataset):

			break

	D = D / (num * 5)

	D = math.sqrt(D)

	print(D) 

	return num, M, D




if __name__ == '__main__':

	# 计算预测时间片与当前时间片的最大插值

	# max_time_diff = trafficData()

	# print(max_time_diff)

	
	# n,m,d = getMD(19.444444)

	stateVspeed()

	# getMDn()

	