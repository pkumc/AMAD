#!/usr/bin/python
# -*- coding:  utf-8 -*- 

'''
Data pre process for anomaly detection

@author: 
Zheng Gao (gao27@indiana.edu) 
'''
from __future__ import print_function
import os
import numpy as np
import math  
class LoadData(object):
	'''
 	make sure the item index starts from 1. Because 0 means the dummy node for padding
	the input tensor is already converted to id
	'''
	def __init__(self,datafile): 
		self.datafile = datafile  
		self.feature_num = self.get_feature_num()# a number. get number of features 
		print("----------finish calculating feature_num!-----------")
		self.feature_dim,self.feature_item_num = self.get_maximum_length() # two lists. feature_dim is a list of all features dimension. feature_item_num is the number of uniqe items in each feature
		print("----------finish calculating feature_dim and feature_item_num!-----------")
		
		#these are the two inputs of our model
		self.feature_index = self.get_feature_index()
		print("----------finish calculating feature_index!-----------")
		
		self.data,self.label,self.anomaly_num = self.padding_data()
		print("----------finish generating padding data!-----------")
		
	def get_feature_num(self):# get the number of features for each instance
		with open(self.datafile) as f:	
			for line in f:
				result = line.rstrip().split(" ")
				feature_num = len(result)
				break
		return feature_num - 1 #the first column refers to the instance label. So remove it.

	#get the maximum dimension in each feature, so as to pad all instances in the same dimension
	def get_maximum_length(self):# 
		feature_dim = [0 for i in range(self.feature_num)] # the largest number of ids in an instance, it will be used as the feature dimension
		feature_item_num = [0 for i in range(self.feature_num)]# number of ids in each feature
		item_id_set = set()
		with open(self.datafile) as f:	
			for line in f:
				features = line.rstrip().split(" ") 
				features.pop(0) # pop the first column,the first column is the instance label, so ignore it.
				for i in range(len(features)):
					items = features[i].split(",")
					curr_feature_dim = len(items)
					if curr_feature_dim > feature_dim[i]:
						feature_dim[i] = curr_feature_dim
					for item in items: 
						item_id_set.add(item)
						if int(item) > feature_item_num[i]:#get the largest item id
							feature_item_num[i] = int(item)
						
		for i in range(len(feature_item_num)): # because index contains 0.
			feature_item_num[i] = feature_item_num[i] + 1
		return feature_dim, len(item_id_set)+1

	#each instance is a vector of all features' concatenation. This function returns the beginning index of each feature
	def get_feature_index(self):
		'''
		eg. if the data feature dimensions are (2,1,3). the constructed feature index is :
		[0,2,3,6]
		'''
		feature_index = [0]
		for i in range(len(self.feature_dim)):
			feature_index.append(self.feature_dim[i]+feature_index[i])
		return feature_index

	#return constructed data
	def padding_data(self):
		data = []
		label = []
		anomaly_num = 0
		with open(self.datafile) as f:	
			for line in f:
				instance = []
				features = line.rstrip().split(" ") 
				label.append(int(features[0]))# append label

				if int(features[0]) == 0:
					anomaly_num = anomaly_num + 1
				features.pop(0) # pop the first column,the first column is the instance label, so ignore it.
				for i in range(len(features)): 
					items = features[i].split(",")
					for j in range(len(items)):
						instance.append(int(items[j]))
					for j in range(len(items),self.feature_dim[i]):
						instance.append(0)
				data.append(instance)
		data = np.asarray(data)
		return data,label,anomaly_num

#convert data to batched data
def get_shaped_data(data,batch_size,block_size,feature_dim):
	data_num = len(data) - len(data)%(block_size*batch_size)
	batch_num = len(data)//(block_size*batch_size)
	# print(data_num)
	data = data[0:data_num,]
	data = np.reshape(data,[batch_num,batch_size,block_size,feature_dim])
	return data
 

if __name__ == '__main__':
	path = "/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy/format.txt" 
	dataset = LoadData(path)
	data = dataset.data
	label = dataset.label
	anomaly_num = dataset.anomaly_num
	index = dataset.feature_index
	feature_item_num = dataset.feature_item_num
	print(data,index,feature_item_num)
	data = get_shaped_data(data,1,2,len(data[0]))
	print(data,data.shape,label,anomaly_num)

	'''
	input:
	1 2 3 0
	1 3,4,2 1 5,3
	1 1,2 2 1,2,4
	1 1,2 2 1,2,4
	1 1,2 2 1,2,4
	0 1,2 2 1,2,4

	output:
	data:
	[[2 0 0 3 0 0 0]
	 [3 4 2 1 5 3 0]
	 [1 2 0 2 1 2 4]
	 [1 2 0 2 1 2 4]
	 [1 2 0 2 1 2 4]
	 [1 2 0 2 1 2 4]] [0, 3, 4, 7]
	batched_data
	[[[[2 0 0 3 0 0 0]
	   [3 4 2 1 5 3 0]]

	  [[1 2 0 2 1 2 4]
	   [1 2 0 2 1 2 4]]]] (1, 2, 2, 7)
	'''










