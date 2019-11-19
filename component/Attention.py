#!/usr/bin/python
# -*- coding:  utf-8 -*-
'''
Tensorflow implementation of attention to get the feature level importance score

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

'''
input: feature vectors with 4-D dimension: feature_size, batch_size, block_size, instance_dim
output: instance vectors with 3-D dimensions: batch_size, block_size, instance_dim
reference1: https://github.com/MachineLP/train_cnn-rnn-attention/blob/master/net/attention/attention.py
reference2: https://blog.csdn.net/qsczse943062710/article/details/79539005
'''
class Attention(object):
	def  __init__(self,attention_dim,feature_size,block_size,instance_dim,random_seed=2018): 
		self.attention_dim = attention_dim
		self.feature_size = feature_size
		self.block_size = block_size
		self.instance_dim = instance_dim
		self.random_seed = random_seed

		self.data = tf.placeholder(tf.float32, shape=[self.feature_size,None,self.block_size,self.instance_dim], name="data") 
			
		with tf.variable_scope("attention"): 
			self.att_v = tf.get_variable('vector',initializer= tf.ones([1,self.attention_dim]), dtype=tf.float32) 
			self.att_w = tf.get_variable('weight',initializer= tf.ones([self.instance_dim,self.attention_dim]), dtype=tf.float32) 
			self.att_b = tf.get_variable('bias',initializer= tf.ones([1,self.attention_dim]), dtype=tf.float32) 

		tf.set_random_seed(self.random_seed) 
		#Input data
		
		data_transpose = tf.transpose(self.data,[1,2,0,3]) # turn to batch_size, block_size, feature_size, instance_dim
		weight = tf.tile(tf.reshape(self.att_w,[1,1,tf.shape(self.att_w)[0],tf.shape(self.att_w)[1]]),[tf.shape(data_transpose)[0],tf.shape(data_transpose)[1],1,1]) 
		bias = tf.tile(tf.reshape(self.att_b,[1,1,tf.shape(self.att_b)[0],tf.shape(self.att_b)[1]]),[tf.shape(data_transpose)[0],tf.shape(data_transpose)[1],self.feature_size,1])
		vector = tf.tile(tf.reshape(self.att_v,[1,1,tf.shape(self.att_v)[0],tf.shape(self.att_v)[1]]),[tf.shape(data_transpose)[0],tf.shape(data_transpose)[1],1,1])
		u= tf.tanh(tf.matmul(data_transpose,weight)+bias)
		vu = tf.matmul(u, tf.transpose(vector, [0,1,3,2]))  
		exp = tf.exp(vu)
		exp_sum = tf.reduce_sum(exp, 2) 
		shape = tf.shape(exp_sum)
		exp_sum = tf.reshape(exp_sum, [shape[0],shape[1],shape[2],-1]) 
		alphas = exp / exp_sum

		self.instance_vector = tf.reduce_sum(data_transpose*alphas,2)
		

		self._init_session()

	def _init_session(self): 
		# adaptively growing video memory
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True 
		self.sess = tf.Session(config=config)
		# self.saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		self.sess.run(init)
		
	def train(self,data):

		self.graph = tf.Graph()
		with self.graph.as_default():   
			feed_dict = {self.data: data}
			data = self.sess.run((self.data,self.instance_vector), feed_dict=feed_dict)
			print("result is:",data[0],data[0].shape,data[1],data[1].shape)#data[2],data[2][0].shape
		
	def close(self):#shut down the session
		self.sess.close()

if __name__ == '__main__': 
	f1 = [[[1,1,1,1,1],[2,2,2,2,2]],
			[[3,3,3,3,3],[4,4,4,4,4]]]
	f2 = [[[5,5,5,5,5],[6,6,6,6,6]],
			[[7,7,7,7,7],[8,8,8,8,8]]]
	f3 = [[[9,9,9,9,9],[10,10,10,10,10]],
			[[11,11,11,11,11],[12,12,12,12,12]]]

	data = [f1,f2,f3]
	data = np.asarray(data)
	print(data,data.shape,len(data),len(data[0][0]),len(data[0][0][0]))

	model = Attention(10,len(data),len(data[0][0]),len(data[0][0][0]))
	model.train(data)
	model.close()



