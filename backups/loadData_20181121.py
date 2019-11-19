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
from Config import Config
class LoadData(object):
	'''
	given the path of data, return the data format 
	:param datafile,block_size
	return:
	data: a dictionary, each key refers to a feature. And a feature is either a key:value pair or a list of key:value pairs
	'''
	def __init__(self,datafile,features_num,block_size,batch_size): 
		self.datafile = datafile
		self.block_size = block_size
		self.batch_size = batch_size
		self.features_num = features_num # number of features 
		self.features = self.map_features( ) # feature list of dictionary. Each feature is a dictionary of name-id. all features form a list
		self.data = self.read_data( ) # stored as a list of lists
		# print(type(self.data),len(self.data),self.data[0])
		self.data = self.get_block_data()
		# print(type(self.data),len(self.data),self.data[0])
		self.batch_data = self.get_batch_data() 
	def map_features(self):
		features = self.initialize_features()  
		self.feature_label = self.read_features(features) 
		print("features number: %d, and there are %d different ids in the first feature" % (len(features),len(features[0])))
		# print(features)
		return  features

	def initialize_features(self): # get to know the number of features 
		features = [{} for i in range(self.features_num)]    
		return features

	def read_features(self, features): # read a feature file
		feature_label = [0 for x in range(self.features_num)] # 0 refers that the feature is id only. 1 means the feature is id list
		f = open( self.datafile )
		line = f.readline() 
		while line:
			line = line.strip().split(' ')  
			for i in range(len(line)):  
				l = line[i] # each feature
				items = l.strip().split(',')  
				if len(items) > 1:
					feature_label[i] = 1
				for item in items: 
					item = item.split(":")[0]
					if item not in features[i]: 
						features[i][ item ] = len(features[i])
			line = f.readline()
		f.close()
		# print(features)
		return feature_label

	def read_data(self):
		# read a data file. For a row, the first column goes into Y_;
		# the other columns become a row in X_ and entries are maped to indexs in self.features
		f = open( self.datafile )
		X_ = [] 
		line = f.readline()
		while line:
			line = line.strip().split(' ') 
			temp = [[] for i in range(len(self.features))] # store each instance
			
			for i in range(len(line)): # the rest features convert to id
				l = line[i] # each feature
				items = l.strip().split(',')  
				for item in items: 
					item_name = item.split(':')[0]  
					item_id = self.features[i][item_name] 
					temp[i].append(item_id) 
			X_.append(temp)		
			line = f.readline()
		f.close() 
		return X_ 

	def get_block_data(self):
		block_num = math.floor(len(self.data)/self.block_size)
		# print("block_num",block_num)
		block_data = []
		for i in range(block_num):
			if (i+1)*self.block_size > len(self.data): # for the last instances which can't form a complete block, throw it
				continue  
			# print("block id",i)
			temp = []
			for j in range(i*self.block_size,(i+1)*self.block_size): 
				# print("data id",j)
				temp.append(self.data[j])
			# break
			block_data.append(temp)	
		# processint("block_data",len(block_data))	   

		return block_data

	def get_batch_data(self):
		batch_num = math.floor(len(self.data)/self.batch_size)
		# print("block_num",block_num)
		batch_data = []
		for i in range(batch_num):
			if (i+1)*self.batch_size > len(self.data): # for the last instances which can't form a complete batch, throw it
				continue  
			# print("block id",i)
			temp = []
			for j in range(i*self.batch_size,(i+1)*self.batch_size): 
				# print("data id",j)
				temp.append(self.data[j])
			# break
			batch_data.append(temp)	
		return batch_data
if __name__ == '__main__':
	path = "/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/synthetic/sample.txt" 
	dataset = LoadData(path,Config.feature_num,Config.block_size,Config.batch_size)
	d = dataset.batch_data
	f = dataset.feature_label
	print(type(d),len(d),d[0]) 
	print(f)














