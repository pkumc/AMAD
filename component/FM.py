#!/usr/bin/python
# -*- coding:  utf-8 -*-
'''
Tensorflow implementation to learn instance vector representation

@author: 
zheng gao

@references:
'''
from __future__ import print_function
import math
import os
import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from time import time 
import loadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm 

#################### Arguments ####################
class FM(object):
	# def __init__(self, label, feature_num, block_size, batch_size, learning_rate, keep,
	# 	optimizer_type, batch_norm,random_seed=2018):
	# 	# bind params to class
	# 	self.label = label
	# 	self.feature_num = feature_num
	# 	self.block_size = block_size
	# 	self.batch_size = batch_size
	# 	self.learning_rate = learning_rate
	# 	self.keep = keep
	# 	self.optimizer_type = optimizer_type
	# 	self.batch_norm = batch_norm 
	# 	self.random_seed = random_seed
	def __init__(self, feature_index,FM_weight_dim, FM_each_feature_size,batch_size, block_size,instance_dim, random_seed=2018):
		self.feature_index = feature_index # the beginning index of each feature in the concatenate feature vector, starts with 0 and ends with the concatenate feature vector length
		self.FM_weight_dim = FM_weight_dim # the dimension of the weight embeddings
		self.FM_each_feature_size = FM_each_feature_size # the dimension (number of id) in each feature, stored in a list
		self.batch_size = batch_size 
		self.block_size = block_size
		self.instance_dim = instance_dim # the dimension of the concatenated feature, which is the input
		self.random_seed = random_seed
		# init all variables in a tensorflow graph
		self._init_graph() 

	def _init_graph(self):
		
		# Set graph level random seed
		tf.set_random_seed(self.random_seed)

		#Input data
		self.data = tf.placeholder(tf.int32, shape=[None,self.block_size,self.instance_dim], name="train_features") 
		# self.feature_label = tf.placeholder(tf.int32,name="feature_label")  
		# self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep_fm")
		
		# initialize all weight emebddings 
		self.feature_weight_embedding_list = [] 
		with tf.variable_scope("FM"): 
			for i in range(len(self.FM_each_feature_size)):
				first_order_weight = tf.get_variable('weight_1_embeddings_feature_{}'.format(i),initializer= tf.ones([self.FM_each_feature_size[i],self.FM_weight_dim]), dtype=tf.float32) 
				second_order_weight = tf.get_variable('weight_2_embeddings_feature_{}'.format(i),initializer= tf.ones([self.FM_each_feature_size[i],self.FM_weight_dim]), dtype=tf.float32) 
				self.feature_weight_embedding_list.append((first_order_weight,second_order_weight))# form a tuple and append to the list

		# segment the concatenated feature into several individual features
		self.tensor_list = []
		for i in range(len(self.feature_index)-1):
			temp = tf.slice(self.data,[0,0,self.feature_index[i]],[tf.shape(self.data)[0],tf.shape(self.data)[1],self.feature_index[i+1]-self.feature_index[i]])
			self.tensor_list.append(temp)
		
		#calculate FM feature level representation, in each feature
		self.feature_vectors = []
		for i in range(len(self.tensor_list)):
			sliced_feature = self.tensor_list[i]  #[batch_size,block_size,feature_size]
			related_weight_first_order = self.feature_weight_embedding_list[i][0]
			related_weight_second_order = self.feature_weight_embedding_list[i][1]

			'''
			calculate first order vector: sum all id vectors together, and substract all id=0 vectors 
			'''

			first_order_vec = tf.nn.embedding_lookup(related_weight_first_order,sliced_feature) # 4-D
			sliced_feature_first = tf.clip_by_value(sliced_feature, 0, 1) # convert all eligible ids to 1, keeps the dummy id 0 as 0
			sliced_feature_first = tf.expand_dims(sliced_feature_first, -1) # expend one more dimension in the last dimension 

			first_order_vec = tf.multiply(first_order_vec,tf.cast(sliced_feature_first,tf.float32)) # dot product to get eligible id vectors. dummy id 0 has a vector 0
			first_order_vec = tf.reduce_sum(first_order_vec,axis=2,name='fm_first') #3-D, sum on the third dimension. sum up all id vectors in the same instance
			
			# self.first_order_vec = first_order_vec

			'''
			calculate second order vector:
			'''
			second_order_vec = tf.nn.embedding_lookup(related_weight_second_order,sliced_feature) # 4-D
			sliced_feature_second = tf.clip_by_value(sliced_feature, 0, 1) # convert all eligible ids to 1, keeps the dummy id 0 as 0
			sliced_feature_second = tf.expand_dims(sliced_feature_second, -1) # expend one more dimension in the last dimension 

			second_order_vec = tf.multiply(second_order_vec,tf.cast(sliced_feature_second,tf.float32)) # dot product to get eligible id vectors. dummy id 0 has a vector 0
			
			#sum square
			second_order_vec_sum = tf.reduce_sum(second_order_vec,axis=2) #3-D, sum on the third dimension. sum up all id vectors in the same instance
			second_order_sum_square = tf.square(second_order_vec_sum)# (a+b)^2

			#square sum
			second_order_vec_square = tf.square(second_order_vec)
			second_order_square_sum = tf.reduce_sum(second_order_vec_square,axis=2)# a^2+b^2

			#second order vector
			second_order_vec = 0.5 * tf.subtract(second_order_sum_square,second_order_square_sum,name='fm_second')
			# self.second_order_vec = second_order_vec
			

			# concatenate two first order and second order feature together 
			current_feature_vec = tf.concat([first_order_vec,second_order_vec],-1)
			# self.current_feature_vec = current_feature_vec
			self.feature_vectors.append(current_feature_vec)
			# break
		self.feature_vectors = tf.stack(self.feature_vectors,axis = 0) # stack all feature vectors together to form a uniformed tensor
		self._init_session() # initilize all parameters in this default graph

	def _init_session(self): 
		# adaptively growing video memory
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True 
		self.sess = tf.Session(config=config)
		# self.saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		self.sess.run(init)
		

	def close(self):#shut down the session
		self.sess.close()

	def batch_fit(self, data):  # fit a batch
		# feed_dict = {self.data: data, self.feature_label:self.label,self.dropout_keep: self.keep}
		feed_dict = {self.data: data}
		data = self.sess.run((self.data,self.feature_vectors), feed_dict=feed_dict)
		print("batch is:",data[1],data[1].shape)#data[2],data[2][0].shape
		# loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
		# return loss

	def train(self, data): #80% for training, 10% for validation and 10% for testing
		total_batch_num = len(data)//self.batch_size
		print("total_batch_num",total_batch_num)
		self.graph = tf.Graph()
		with self.graph.as_default():   
			for i in range(total_batch_num):
				train_begin_index = i*self.batch_size
				validate_begin_index = i*self.batch_size + int(math.ceil(self.batch_size*0.8)) # 
				test_begin_index = i*self.batch_size + int(math.ceil(self.batch_size*0.9))
				print("train_begin_index",train_begin_index,"validate_begin_index",validate_begin_index,"test_begin_index",test_begin_index)


				batch_train = data[train_begin_index:validate_begin_index,]
				batch_validate = data[validate_begin_index:(i+1)*self.batch_size,]
				batch_test = data[validate_begin_index:(i+1)*self.batch_size,] # here we make the validate and the test data are the same
				
				self.batch_fit(batch_train)	# fit the model

	def evaluate(self, data):
		pass
'''
this function is used to train the model
'''
def train(path):
	# dataset = LoadData(path,Config.feature_num,Config.block_size,Config.batch_size)
	# data = dataset.batch_data
	# feature_label = dataset.feature_label # specify id and id-list
	data = [[[2,0,1,1,2],[1,2,1,1,0]],
			[[1,0,0,3,2],[0,0,2,0,0]]]
	data = np.asarray(data)
	feature_index = [0,2,3,len(data[0][0])] 
	FM_each_feature_size = [3,3,5]
	FM_weight_dim = 5
	print(data,data.shape,type(data))

	batch_size = 2
	block_size = 2
	instance_dim = len(data[0][0])
	model = FM(feature_index,FM_weight_dim,FM_each_feature_size,batch_size,block_size,instance_dim)
	model.train(data)
	model.close()
if __name__ == '__main__':
	path = "/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/synthetic/sample.txt" 
	train(path)


