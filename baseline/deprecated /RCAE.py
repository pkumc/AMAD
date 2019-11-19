#!/usr/bin/python
# -*- coding:  utf-8 -*- 
'''
robust deep and inductive anomaly detection 
https://github.com/raghavchalapathy/rcae
Configuration: parameter settings used in the whole model
Model: graph structure 
@author: 
Zheng Gao (gao27@indiana.edu) 
'''
from __future__ import print_function 
import tflearn
import numpy as np
from sklearn import metrics
import tensorflow as tf 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import loadData as ld

def parse_args():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
	parser.add_argument('--input', default='/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy/format.txt', help='data path')
	parser.add_argument('--batch-size', type=int, default=10, help='batch size')
	parser.add_argument('--block-size', type=int, default=10, help='block size')
	parser.add_argument('--hidden-dim', type=int, default=128, help='hidden dimension number')
	parser.add_argument('--epoch', type=int, default=10, help='epoch number')
	parser.add_argument('--block-ratio', type=float, default=0.9, help='gamma for loss')
	parser.add_argument('--threshold-scale', type=float, default=2.0, help='threshold to detect anomaly')
	parser.add_argument('--categorical', dest='categorical', action='store_true')
	parser.add_argument('--no-categorical', dest='no-categorical', action='store_false')
	parser.set_defaults(categorical=True)
	args = parser.parse_args()
	return args

class Config(object): 
	def __init__(self,args):
		self.batch_size = args.batch_size
		self.block_size = args.block_size
		self.hidden_dim = args.hidden_dim
		self.epoch = args.epoch
		self.block_ratio = args.block_ratio
		self.threshold_scale = args.threshold_scale

class RCAE(object): 

	def __init__(self,instance_dim,hidden_dim):
		
		self.instance_dim = instance_dim
		self.hidden_dim = hidden_dim 
		
		hidden_layer = None
		decode_layer = None
		# Building the autoencoder model
		net = tflearn.input_data(shape=[None,self.instance_dim], name="data")
		net = tflearn.reshape(net,[-1,1,1,net.shape[1]])#turn to 4 D
		[net,hidden_layer] = self.encoder(net,hidden_layer)
		[net,decode_layer] = self.decoder(net,decode_layer)
		mue = 0.1
		net = tflearn.regression_RobustAutoencoder(net,mue,hidden_layer,decode_layer, optimizer='adam', learning_rate=0.001,
						loss='rPCA_autoencoderLoss', metric=None,name="vanilla_autoencoder")
		#rPCA_autoencoderLoss_FobsquareLoss
		#rPCA_autoencoderLoss
		#net = tflearn.regression(net, optimizer='adam', loss='mean_square', metric=None)
		model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='tensorboard/')
	
	# Define the convoluted ae architecture
	def encoder(self,inputs,hidden_layer):
		net = tflearn.conv_2d(inputs, 16, 3, strides=2)
		net = tflearn.batch_normalization(net)
		net = tflearn.elu(net)
		print "========================"
		print "enc-L1",net.get_shape()
		print "========================"

		net = tflearn.conv_2d(net, 16, 3, strides=1)
		net = tflearn.batch_normalization(net)
		net = tflearn.elu(net)
		print "========================"
		print "enc-L2",net.get_shape()
		print "========================"

		net = tflearn.conv_2d(net, 32, 3, strides=2)
		net = tflearn.batch_normalization(net)
		net = tflearn.elu(net)
		print "========================"
		print "enc-L3",net.get_shape()
		print "========================"
		net = tflearn.conv_2d(net, 32, 3, strides=1)
		net = tflearn.batch_normalization(net)
		net = tflearn.elu(net)
		print "========================"
		print "enc-L4",net.get_shape()
		print "========================"
		net = tflearn.flatten(net)
		#net = tflearn.fully_connected(net, nb_feature,activation="sigmoid")
		net = tflearn.fully_connected(net, self.instance_dim)
		hidden_layer = net
		net = tflearn.batch_normalization(net)
		net = tflearn.sigmoid(net)
		print "========================"
		print "hidden",net.get_shape()
		print "========================"

		return [net,hidden_layer]

	def decoder(self,inputs,decode_layer):
		net = tflearn.fully_connected(inputs, self.hidden_dim, name='DecFC1')
		net = tflearn.batch_normalization(net, name='DecBN1')
		net = tflearn.elu(net)
		print "========================"
		print "dec-L1",net.get_shape()
		print "========================"

		net = tflearn.reshape(net, (-1, 1, 1, self.hidden_dim))
		net = tflearn.conv_2d(net, 32, 3, name='DecConv1')
		net = tflearn.batch_normalization(net, name='DecBN2')
		net = tflearn.elu(net)
		print "========================"
		print "dec-L2",net.get_shape()
		print "========================"
		net = tflearn.conv_2d_transpose(net, 16, 3, [1, self.hidden_dim],
		                                    strides=2, padding='same', name='DecConvT1')
		net = tflearn.batch_normalization(net, name='DecBN3')
		net = tflearn.elu(net)
		print "========================"
		print "dec-L3",net.get_shape()
		print "========================"
		net = tflearn.conv_2d(net, 16, 3, name='DecConv2')
		net = tflearn.batch_normalization(net, name='DecBN4')
		net = tflearn.elu(net)
		print "========================"
		print "dec-L4",net.get_shape()
		print "========================"
		net = tflearn.conv_2d_transpose(net, 1, 3, [1, self.hidden_dim],
		                                    strides=2, padding='same', activation='sigmoid',
		                                    name='DecConvT2')
		decode_layer = net
		print "========================"
		print "output layer",net.get_shape()
		print "========================"
		return [net,decode_layer]

#classification evaluation
def eval(truth,pred):
	evaluation_dict = {}
	acc = metrics.accuracy_score(truth,pred);evaluation_dict['acc']=acc
	precision = metrics.precision_score(truth,pred);evaluation_dict['precision']=precision
	recall = metrics.recall_score(truth,pred);evaluation_dict['recall']=recall
	f1_macro = metrics.f1_score(truth,pred, average='macro');evaluation_dict['f1_macro']=f1_macro
	f1_micro = metrics.f1_score(truth,pred, average='micro');evaluation_dict['f1_micro']=f1_micro
	# auc = metrics.roc_auc_score(truth,pred);evaluation_dict['auc']=auc
	return evaluation_dict


def run(args): 
	#load configuration
	config = Config(args) 
	#load data
	path = args.input
	if args.categorical:
		training_data,training_label,testing_data,testing_label = ld.load_categorical(path)
	else:
		training_data,training_label,testing_data,testing_label = ld.load_numberic(path) 
	instance_dim = len(training_data[0])

	with tf.Graph().as_default(),tf.Session() as sess:

		model = RCAE(instance_dim,
						config.hidden_dim)
	
		init = tf.global_variables_initializer()  
		sess.run(init)
		
		for epoch in range(config.epoch):
			#training
			batch_num = len(training_data)//config.batch_size
			for i in range(batch_num):
				curr_batch = training_data[i*config.batch_size:(i+1)*config.batch_size]	
				feed_dict = {model.data: curr_batch}
				result = sess.run((model.D_train,model.G_train,model.average_loss),feed_dict=feed_dict)
				loss_threshold = result[2]
				if i % 100 == 0:
					# if threshold < loss_threshold:
					# 	loss_threshold = threshold
					print("In epoch %d and batch %d, average loss: %.4f "%(epoch,i,loss_threshold))
					
	