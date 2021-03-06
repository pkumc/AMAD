#!/usr/bin/python
# -*- coding:  utf-8 -*- 
'''
Anomaly Detection using One-Class Neural Networks 
https://github.com/raghavchalapathy/oc-nn 
Configuration: parameter settings used in the whole model
Model: graph structure 
@author: 
Zheng Gao (gao27@indiana.edu) 
'''
from __future__ import print_function
import numpy as np
from sklearn import metrics
import tensorflow as tf 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import loadData as ld
import random

def parse_args():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
	parser.add_argument('--input', default='/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy/format.txt', help='data path')
	parser.add_argument('--instance-output', default='/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy/instance_output.txt', help='data path')
	parser.add_argument('--block-output', default='/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy/block_output.txt', help='data path')
	parser.add_argument('--batch-size', type=int, default=10, help='batch size')
	parser.add_argument('--block-size', type=int, default=10, help='block size')
	parser.add_argument('--hidden-dim', type=int, default=128, help='hidden dimension number')
	parser.add_argument('--epoch', type=int, default=10, help='epoch number')
	parser.add_argument('--v', type=float, default=0.5, help='alpha for loss')
	parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate')
	parser.add_argument('--block-ratio', type=float, default=0.9, help='gamma for loss')
	parser.add_argument('--threshold-scale', type=float, default=2.0, help='threshold to detect anomaly')
	parser.add_argument('--categorical', dest='categorical', action='store_true')
	parser.add_argument('--no-categorical', dest='categorical', action='store_false')
	parser.set_defaults(categorical=True)
	args = parser.parse_args()
	return args
class Config(object): 
	def __init__(self,args):
		self.batch_size = args.batch_size
		self.block_size = args.block_size
		self.hidden_dim = args.hidden_dim
		self.epoch = args.epoch
		self.learning_rate = args.learning_rate
		self.v = args.v
		self.block_ratio = args.block_ratio
		self.threshold_scale = args.threshold_scale

class OCNN(object): 

	def __init__(self,instance_dim,hidden_dim,v,learning_rate):
		self.instance_dim = instance_dim
		self.hidden_dim = hidden_dim
		self.v = v 
		self.learning_rate = learning_rate

		self.data =  tf.placeholder(tf.float32, shape=[None,self.instance_dim], name="data")

		self.r = tf.get_variable('boundary_r',initializer= tf.truncated_normal([1]), dtype=tf.float32) 
		 
		with tf.variable_scope("encoder"):
			self.w_enc = tf.get_variable('weight_encoder',initializer= tf.truncated_normal([self.instance_dim,self.hidden_dim]), dtype=tf.float32) 
			self.b_enc = tf.get_variable('bias_encoder',initializer= tf.truncated_normal([self.hidden_dim]), dtype=tf.float32) 

			
		self.output = tf.nn.sigmoid(tf.matmul(self.data,self.w_enc)+self.b_enc)

		with tf.variable_scope("transform"):
			self.w_transform = tf.get_variable('w',initializer= tf.truncated_normal([self.hidden_dim]), dtype=tf.float32) 
			self.b_transform = tf.get_variable('b',initializer= tf.truncated_normal([1]), dtype=tf.float32) 

			
		# self.W = tf.matmul(self.output,self.w_transform)+self.b_transform
		
		self.average_loss = 0.5* tf.reduce_sum(tf.square(self.w_transform)) + 0.5* tf.reduce_sum(tf.square(self.w_enc))+ (1.0/self.v)*tf.reduce_mean(tf.nn.relu(tf.reduce_max(tf.subtract(self.r,self.w_transform*self.output),axis=1)))- self.r
		self.total_loss =  0.5* tf.reduce_sum(tf.square(self.w_transform)) + 0.5* tf.reduce_sum(tf.square(self.w_enc))+ (1.0/self.v)*tf.nn.relu(tf.reduce_max(tf.subtract(self.r,self.w_transform*self.output),axis=1))- self.r
		self.total_loss = tf.reshape(self.total_loss,[-1])
		self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.average_loss,name='train_op')

#classification evaluation
def eval(truth,pred):
	evaluation_dict = {}
	acc = metrics.accuracy_score(truth,pred);evaluation_dict['acc']=acc
	precision = metrics.precision_score(truth,pred,pos_label=0);evaluation_dict['precision']=precision
	recall = metrics.recall_score(truth,pred,pos_label=0);evaluation_dict['recall']=recall
	f1_macro = metrics.f1_score(truth,pred, average='macro',pos_label=0);evaluation_dict['f1_macro']=f1_macro
	f1_micro = metrics.f1_score(truth,pred, average='micro',pos_label=0);evaluation_dict['f1_micro']=f1_micro
	mcc = metrics.matthews_corrcoef(truth,pred);evaluation_dict['mcc']=mcc
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
		training_data,training_label,testing_data,testing_label = ld.load_numeric(path) 
	instance_dim = len(training_data[0])
	#shuffle testing data,to ensure testing data and label are shuffled in the same way
	randnum = random.randint(0,100)
	random.seed(randnum)
	random.shuffle(testing_data)
	random.seed(randnum)
	random.shuffle(testing_label)
	with tf.Graph().as_default(),tf.Session() as sess:

		model = OCNN(instance_dim,
						config.hidden_dim,
						config.v,
						config.learning_rate)
	
		init = tf.global_variables_initializer()  
		sess.run(init)
		
		
		for epoch in range(config.epoch):
			#training
			batch_num = len(training_data)//config.batch_size
			for i in range(batch_num):
				curr_batch = training_data[i*config.batch_size:(i+1)*config.batch_size]	
				feed_dict = {model.data: curr_batch}
				result = sess.run((model.train_op,model.average_loss),feed_dict=feed_dict)
				loss_threshold = result[1]
				print("loss:",loss_threshold,loss_threshold.shape)
				if i % 100 == 0:
					# if threshold < loss_threshold:
					# 	loss_threshold = threshold
					print("In epoch %d and batch %d, average loss: %.4f "%(epoch,i,loss_threshold))
				
			'''
			testing after each epoch
			'''
			#individual instance level evaluation
			loss_threshold = loss_threshold * config.threshold_scale
			feed_dict = {model.data: testing_data}
			testing_data_loss = sess.run(model.total_loss,feed_dict=feed_dict)
			# print("testing_data_loss",testing_data_loss,testing_data_loss.shape)
			individual_pred = []
			for i in range(len(testing_data_loss)):
				if testing_data_loss[i] < loss_threshold:
					individual_pred.append(1)
				else:
					individual_pred.append(0)
			print(len(individual_pred),len(testing_label))
			instance_eval = eval(testing_label,individual_pred)		

			#write instance loss to file
			bw = open(args.instance_output, 'w')
			bw.write("true pred\n")
			for i in range(len(testing_data_loss)):
				bw.write(str(testing_label[i])+ " "+str(testing_data_loss[i])+"\n")
			bw.close()

			#block level evaluation
			pred_block = []
			true_block = []
			testing_block_num = len(testing_data) // config.block_size
			for i in range(testing_block_num):
				pred_sum = np.sum(individual_pred[i*config.block_size:(i+1)*config.block_size])
				true_sum = np.sum(testing_label[i*config.block_size:(i+1)*config.block_size])
				if pred_sum < config.block_size*config.block_ratio:
					pred_block.append(0)
				else:
					pred_block.append(1)
				if true_sum < config.block_size*config.block_ratio:
					true_block.append(0)
				else:
					true_block.append(1)
			block_eval = eval(true_block,pred_block)

			#write block loss to file
			bw = open(args.block_output, 'w')
			bw.write("true pred\n")
			for i in range(testing_block_num):
				true_block = "1"
				pred_block = np.mean(testing_data_loss[i*config.block_size:(i+1)*config.block_size])
				true_sum = np.sum(testing_label[i*config.block_size:(i+1)*config.block_size])
				if true_sum < config.block_size*config.block_ratio:
					true_block = "0"
				bw.write(true_block+ " "+str(pred_block)+"\n")
			bw.close()
			print("instance level evaluation: ",instance_eval)
			print("block level evaluation: ",block_eval)				
if __name__ == '__main__':
	args = parse_args()
	run(args)
		





