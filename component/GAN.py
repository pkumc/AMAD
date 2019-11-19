#!/usr/bin/python
# -*- coding:  utf-8 -*-
'''
GAN method for anomaly detection
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


class GAN(object):
	#data is 3-D : [batch_size,block_size,instance_dim]
	# block is 1-D: [block_dim]
	def __init__(self,block_size,instance_dim,block_dim,hidden_dim):
		
		self.block_size = block_size
		self.instance_dim = instance_dim
		self.block_dim = block_dim
		self.hidden_dim = hidden_dim

		self.data_real = tf.placeholder(tf.float32, shape=[None,self.block_size,self.instance_dim], name="data_real") 
		self.data_fake = tf.placeholder(tf.float32, shape=[None,self.block_size,self.instance_dim], name="data_fake") 
		self.block_real = tf.placeholder(tf.float32, shape=[None,self.block_dim], name="block_real") 
		self.block_fake = tf.placeholder(tf.float32, shape=[None,self.block_dim], name="block_fake") 
		

		with tf.variable_scope("instance"): 
			self.w_instance_hidden = tf.get_variable('w1',initializer= tf.ones([self.instance_dim,self.hidden_dim]), dtype=tf.float32)  
			self.b_instance_hidden = tf.get_variable('b1',initializer= tf.ones([1,self.hidden_dim]), dtype=tf.float32) 

			self.w_instance_output = tf.get_variable('w2',initializer= tf.ones([self.hidden_dim,1]), dtype=tf.float32) 
			self.b_instance_output = tf.get_variable('b2',initializer= tf.ones([1,1]), dtype=tf.float32) 

		with tf.variable_scope("block"): 
			self.w_block_hidden = tf.get_variable('w1',initializer= tf.ones([self.block_dim,self.hidden_dim]), dtype=tf.float32) 
			self.b_block_hidden = tf.get_variable('b1',initializer= tf.ones([1,self.hidden_dim]), dtype=tf.float32) 

			self.w_block_output = tf.get_variable('w2',initializer= tf.ones([self.hidden_dim,1]), dtype=tf.float32) 
			self.b_block_output = tf.get_variable('b2',initializer= tf.ones([1,1]), dtype=tf.float32) 

		self.logit_D_real = self.discriminator_instance(self.data_real)
		self.logit_D_fake = self.discriminator_instance(self.data_real)
		self.logit_B_real = self.discriminator_block(self.block_real)
		self.logit_B_fake = self.discriminator_block(self.block_fake)

		D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_D_real, labels=tf.ones_like(self.logit_D_real)))
		D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_D_fake, labels=tf.zeros_like(self.logit_D_fake))) #对判别器对虚假样本(即生成器生成的手写数字)的判别结果计算误差(将结果与0比较)
		B_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_B_real, labels=tf.ones_like(self.logit_B_real)))
		B_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_B_fake, labels=tf.zeros_like(self.logit_B_fake))) #对判别器对虚假样本(即生成器生成的手写数字)的判别结果计算误差(将结果与0比较)
		self.discriminator_loss= D_real_loss+D_fake_loss+B_real_loss+B_fake_loss
		self._init_session()
	def generator(self):
		pass

	def discriminator_instance(self,data):
		instance = tf.reshape(data,[-1,self.instance_dim])
		hidden_layer = tf.nn.relu(tf.matmul(instance, self.w_instance_hidden) + self.b_instance_hidden)
		output = tf.matmul(hidden_layer,self.w_instance_output)+self.b_instance_output
		return output
	def discriminator_block(self,data):
		hidden_layer = tf.nn.relu(tf.matmul(data, self.w_block_hidden) + self.b_block_hidden)
		output = tf.matmul(hidden_layer,self.w_block_output)+self.b_block_output
		return output
	def _init_session(self): 
		# adaptively growing video memory
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True 
		self.sess = tf.Session(config=config)
		# self.saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		self.sess.run(init)
		
	def train(self,data_real,data_fake,block_real,block_fake): 
		self.graph = tf.Graph()
		with self.graph.as_default():   
			feed_dict = {self.data_real: data_real,self.data_fake: data_fake,self.block_real: block_real,self.block_fake: block_fake}
			data = self.sess.run((self.discriminator_loss,self.discriminator_loss), feed_dict=feed_dict)
			print("result is:",data[0],data[0].shape,data[1],data[1].shape)#data[2],data[2][0].shape
		
	def close(self):#shut down the session
		self.sess.close()	

if __name__ == '__main__': 
	data_real = [[[1,1,1,1,1],[2,2,2,2,2]],
			[[3,3,3,3,3],[4,4,4,4,4]]]
	data_fake = [[[1,1,1,1,1],[2,2,2,2,2]],
			[[3,3,3,3,3],[4,4,4,4,4]]]
	block_real = np.asarray([[1,1,1,1]])
	block_fake = np.asarray([[1,1,1,1]])
	print(block_real.shape)
	model = GAN(2,5,4,3)
	model.train(data_real,data_fake,block_real,block_fake)
	model.close()
