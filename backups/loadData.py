'''
Data pre process for anomaly detection

@author: 
Zheng Gao (gao27@indiana.edu) 
'''
import os
import numpy as np

class LoadData(object):
	'''
	given the path of data, return the data format 
	:param path
	return:
	Train_data: a dictionary, each key refers to a feature. And a feature is either a key:value pair or a list of key:value pairs
	Test_data: same as Train_data
	Validation_data: same as Train_data
	'''
	# Three files are needed in the path
	def __init__(self, path,block_size):
		self.path = path + "/"
		self.trainfile = self.path + "training.txt"
		self.testfile = self.path + "testing.txt"
		self.validationfile = self.path + "validation.txt"
		self.features_num = self.map_features( )
		self.Train_data, self.Validation_data, self.Test_data = self.construct_data( ) # stored as a list of lists

	def map_features(self): # map the feature entries in all files, kept in self.features dictionary
		self.features = self.initialize_features(self.trainfile) # first two features don't need to convert to ID
		self.read_features(self.trainfile)
		self.read_features(self.testfile)
		self.read_features(self.validationfile)
		print("features number: %d, and there are %d different ids in the third feature (the real first one)" % (len(self.features),len(self.features[2])))
		return  len(self.features)

	def initialize_features(self,file): # get to know the number of features
		f = open( file )
		line = f.readline()
		num = len(line.strip().split('	')) 
		features = [{} for i in range(num)]  
		f.close()
		return features

	def read_features(self, file): # read a feature file
		f = open( file )
		line = f.readline() 
		id_count = [len(x) for x in self.features] # count how many unique ids already exists in each feature
		while line:
			line = line.strip().split('	')  
			for i in range(2,len(line)): # the first two features are timestamp and CTR tag, which don't need to convert to ID
				l = line[i] # each feature
				items = l.strip().split(',')  
				for item in items: 
					item = item.split(":")[0]
					if item not in self.features[i]:
						feature_len = len(self.features[i]) # number of unique ids in the current feature
						self.features[i][ item ] = feature_len 
			line = f.readline()
		f.close()

	def construct_data(self):
		Train_data = self.read_data(self.trainfile) 
		#print("Number of samples in Train:" , len(Y_))

		Validation_data = self.read_data(self.validationfile) 
		#print("Number of samples in Validation:", len(Y_))

		Test_data = self.read_data(self.testfile) 
		#print("Number of samples in Test:", len(Y_))

		return Train_data,  Validation_data,  Test_data

	def read_data(self, file):
		# read a data file. For a row, the first column goes into Y_;
		# the other columns become a row in X_ and entries are maped to indexs in self.features
		f = open( file )
		X_ = [] 
		line = f.readline()
		while line:
			line = line.strip().split('	') 
			temp = [[] for i in range(len(self.features))] # store each instance

			for i in range(0,2): # the first two features keeps the same
				temp[i].append(float(line[i]))

			for i in range(2,len(line)): # the rest features convert to id
				l = line[i] # each feature
				items = l.strip().split(',')  
				for item in items: 
					item_name = item.split(':')[0]
					item_value = float(item.split(':')[1])
					item_id = self.features[i][item_name] 
					temp[i].append(item_id)
			X_.append(temp)		
			line = f.readline()
		f.close()
		return X_ 

if __name__ == '__main__':
	path = "/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy" 
	data = LoadData(path)
	train = data.Train_data
	print(train)
















