#!/usr/bin/python
# -*- coding:  utf-8 -*- 

'''
Data pre process for anomaly detection baseline
Use this to load aligned data
@author: 
Zheng Gao (gao27@indiana.edu) 
'''
from __future__ import print_function
import os
import numpy as np
import math 

def load_categorical(path):
	training_data = []
	training_label = []
	testing_data = []
	testing_label = []
	distinct_id_set = set()
	with open(path) as f:	
		for line in f:
			result = line.rstrip().split(" ")
			label = result[0]
			for i in range(1,len(result)): 
				features = result[i].split(",")
				for iid in features:
					distinct_id_set.add(int(iid))

	with open(path) as f:	
		for line in f:
			result = line.rstrip().split(" ")
			id_set = set()
			label = int(result[0])
			for i in range(1,len(result)): 
				features = result[i].split(",")
				for iid in features:
					id_set.add(int(iid))		
			encoding = []	
			for i in range(len(distinct_id_set)):
				if i in id_set:
					encoding.append(1)
				else:
					encoding.append(0)
			if label == 1:
				training_label.append(label)
				training_data.append(encoding)
			else:
				testing_label.append(label)#get all anomaly data
				testing_data.append(encoding)#get all anomaly data
	anomaly_len = len(testing_label)
	normal_len = len(training_label)
	#add equal number of normal data to testing data
	testing_data.extend(training_data[normal_len-anomaly_len:normal_len])
	testing_label.extend(training_label[normal_len-anomaly_len:normal_len])
	training_data = training_data[0:normal_len-anomaly_len]
	training_label = training_label[0:normal_len-anomaly_len]

	return training_data,training_label,testing_data,testing_label


def load_numeric(path):
	training_data = []
	training_label = []
	testing_data = []
	testing_label = []
	with open(path) as f:	
		for line in f:
			result = line.rstrip().split(" ")
			label = int(result[0])
			temp = result[1].split(",")
			encoding = []
			for i in range(len(temp)):
				encoding.append(float(temp[i]))
			if label == 1:
				training_label.append(label)
				training_data.append(encoding)
			else:
				testing_label.append(label)#get all anomaly data
				testing_data.append(encoding)#get all anomaly data
	anomaly_len = len(testing_label)
	normal_len = len(training_label)
	#add equal number of normal data to testing data
	testing_data.extend(training_data[normal_len-anomaly_len:normal_len])
	testing_label.extend(training_label[normal_len-anomaly_len:normal_len])
	training_data = training_data[0:normal_len-anomaly_len]
	training_label = training_label[0:normal_len-anomaly_len]
	return training_data,training_label,testing_data,testing_label

if __name__ == '__main__':
	path = "/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/synthetic/synthetic.txt"
	result = load_categorical(path)
	print(len(result[0]),len(result[1]),len(result[2]),len(result[3]))
	# print(result)