#!/usr/bin/python
# -*- coding:  utf-8 -*-
'''
LSTM method to generate block vector given a set of instances
input 3-D vector [batch_size, block_size, instance_dim]
output 2-D vector [batch_size,block_dim], which is block level vector
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
from Config import Config

'''
using instance vectors to learn a block vector via LSTM 
Input: previous block vector and instance vectors in current block
Output: block vector, which is the last hidden state in LSTM
'''
class LSTM(object):
	def  __init__(self,block_size,instance_dim,block_dim,batch_size,random_seed=2018):
		self.block_size = block_size
		self.instance_dim = instance_dim
		self.block_dim = block_dim
		self.batch_size = batch_size# in order for tf.unstack(), we need to offer batch_size here
		self.random_seed = random_seed

		self.data = tf.placeholder(tf.float32, shape=[self.batch_size,self.block_size,self.instance_dim], name="data") 
		#hidden layer is the block vector
		with tf.variable_scope("hidden_layer"): 
			self.block_vector = tf.get_variable('block_vector',initializer= tf.ones([1,self.block_dim]), dtype=tf.float32) 
			self.memory_vector = tf.get_variable('memory_vector',initializer= tf.ones([1,self.block_dim]), dtype=tf.float32) 
		
		tf.set_random_seed(self.random_seed) 

		#create all four gates weights and bias: input (i), forget (f), memory (m) , output (o)

		with tf.variable_scope("input"): 
			self.w_input = tf.get_variable('wi',initializer= tf.ones([self.instance_dim,self.block_dim]), dtype=tf.float32) #sequential instance weight
			self.h_input = tf.get_variable('ui',initializer= tf.ones([self.block_dim,self.block_dim]), dtype=tf.float32) # hidden layer (block vector) weight 
			self.b_input = tf.get_variable('bi',initializer= tf.ones([1,self.block_dim]), dtype=tf.float32) 

		with tf.variable_scope("forget"): 
			self.w_forget = tf.get_variable('wf',initializer= tf.ones([self.instance_dim,self.block_dim]), dtype=tf.float32) #sequential instance weight
			self.h_forget = tf.get_variable('uf',initializer= tf.ones([self.block_dim,self.block_dim]), dtype=tf.float32) # hidden layer (block vector) weight 
			self.b_forget = tf.get_variable('bf',initializer= tf.ones([1,self.block_dim]), dtype=tf.float32) 

		with tf.variable_scope("memory"): 
			self.w_memory = tf.get_variable('wm',initializer= tf.ones([self.instance_dim,self.block_dim]), dtype=tf.float32) #sequential instance weight
			self.h_memory = tf.get_variable('um',initializer= tf.ones([self.block_dim,self.block_dim]), dtype=tf.float32) # hidden layer (block vector) weight 
			self.b_memory = tf.get_variable('bm',initializer= tf.ones([1,self.block_dim]), dtype=tf.float32) 

		with tf.variable_scope("output"): 
			self.w_output = tf.get_variable('wo',initializer= tf.ones([self.instance_dim,self.block_dim]), dtype=tf.float32) #sequential instance weight
			self.h_output = tf.get_variable('uo',initializer= tf.ones([self.block_dim,self.block_dim]), dtype=tf.float32) # hidden layer (block vector) weight 
			self.b_output = tf.get_variable('bo',initializer= tf.ones([1,self.block_dim]), dtype=tf.float32) 

		unstack_batch = tf.unstack(self.data,axis=0) 
		for i in range(len(unstack_batch)):
			unstack_block = tf.unstack(unstack_batch[i],axis=0)
			for j in range(len(unstack_block)):
				current_instance = tf.reshape(unstack_block[j],[-1,tf.shape(unstack_block[j])[0]]) # 2-D: 1*instance_dim

				#input gate 
				input_gate = tf.sigmoid(tf.matmul(current_instance,self.w_input) + tf.matmul(self.block_vector, self.h_input) + self.b_input)

				#forget gate
				forget_gate = tf.sigmoid(tf.matmul(current_instance,self.w_forget) + tf.matmul(self.block_vector, self.h_forget) + self.b_forget)

				#output gate
				output_gate = tf.sigmoid(tf.matmul(current_instance,self.w_output) + tf.matmul(self.block_vector, self.h_output) + self.b_output)

				#memory gate
				memory_gate = tf.tanh(tf.matmul(current_instance,self.w_memory) + tf.matmul(self.block_vector, self.h_memory) + self.b_memory)

				#update
				self.memory_vector = forget_gate * self.memory_vector + input_gate * memory_gate
				self.block_vector = output_gate * tf.tanh(self.memory_vector)
		
		self.output = self.block_vector 

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
			data = self.sess.run((self.data,self.output), feed_dict=feed_dict)
			print("result is:",data[0],data[0].shape,data[1],data[1].shape)#data[2],data[2][0].shape
		
	def close(self):#shut down the session
		self.sess.close()
if __name__ == '__main__': 
	data = [[[1,1,1,1,1],[2,2,2,2,2]],
			[[3,3,3,3,3],[4,4,4,4,4]]]

	model = LSTM(2,5,4,2)
	model.train(data)
	model.close()







