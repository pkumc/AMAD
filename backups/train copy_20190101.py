#!/usr/bin/python
# -*- coding:  utf-8 -*- 

'''
train.py is used to call different models.
We use the next batch data + several generated anomaly data (negative sampling method) to evaluate models optimized until the current batch
@author: 
Zheng Gao (gao27@indiana.edu) 
'''
from __future__ import print_function
import numpy as np
import tensorflow as tf 
from sklearn import metrics
from model import AnomalyNet,Config
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import loadData as ld
import random
def parse_args():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
	parser.add_argument('--input', default='/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy/format.txt', help='data path')
	parser.add_argument('--FM-weight-dim', type=int, default=12, help='FM_weight_dim')
	parser.add_argument('--batch-size', type=int, default=2, help='batch size')
	parser.add_argument('--block-size', type=int, default=2, help='block size')
	parser.add_argument('--attention-dim', type=int, default=12, help='hidden dimension number')
	parser.add_argument('--autoencoder-hidden-dim', type=int, default=12, help='autoencoder dimension number')
	parser.add_argument('--lstm_dropout_keep_prob', type=float, default=1.0, help='drop out rate')
	parser.add_argument('--lstm-layer-num', type=int, default=1, help='LSTM layer number')
	parser.add_argument('--lstm-hidden-size', type=int, default=16, help='LSTM hidden layer dim')
	parser.add_argument('--gan-hidden-dim', type=int, default=12, help='LSTM hidden layer dim')
	parser.add_argument('--epoch', type=int, default=10, help='epoch number')
	parser.add_argument('--alpha', type=float, default=1.0, help='alpha for loss')
	parser.add_argument('--beta', type=float, default=1.0, help='beta for loss')
	parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate')
	parser.add_argument('--instance-confidence', type=float, default=0.9, help='instance confidence for anomaly detection')
	parser.add_argument('--block-ratio', type=float, default=0.5, help='block ratio')
	parser.add_argument('--threshold-scale', type=float, default=1.0, help='threshold to detect anomaly')
	parser.add_argument('--categorical', dest='categorical', action='store_true')
	parser.add_argument('--no-categorical', dest='no-categorical', action='store_false')
	parser.set_defaults(categorical=True)
	args = parser.parse_args()
	return args

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
	dataset = ld.LoadData(args.input)
	data = dataset.data
	label = dataset.label
	anomaly_num = dataset.anomaly_num
	feature_index = dataset.feature_index
	feature_item_num = np.sum(dataset.feature_item_num)
	instance_num = len(data)
	#for training
	training_data = data[:instance_num-2*anomaly_num]
	training_data = ld.get_shaped_data(training_data,config.batch_size,config.block_size,len(data[0]))
	instance_dim = len(training_data[0][0][0])

	#for testing
	testing_data = data[instance_num-2*anomaly_num:]
	testing_label = label[instance_num-2*anomaly_num:]
	#shuffle testing data,to ensure testing data and label are shuffled in the same way
	randnum = config.seed
	random.seed(randnum)
	random.shuffle(testing_data)
	random.seed(randnum)
	random.shuffle(testing_label)

	testing_data = ld.get_shaped_data(testing_data,config.batch_size,config.block_size,len(data[0]))
	testing_data_num = len(testing_label) - len(testing_label)%(config.block_size*config.batch_size)
	testing_label = testing_label[:testing_data_num] # testing data instance level ground truth

	print("instance_dim",training_data.shape,instance_dim)
	print("feature_item_num",feature_item_num)
	with tf.Graph().as_default(),tf.Session() as sess:
		#graph settings
		FM_weight_dim = config.FM_weight_dim
		batch_size = config.batch_size
		block_size = config.block_size
		attention_dim = config.attention_dim
		autoencoder_hidden_dim = config.autoencoder_hidden_dim
		lstm_dropout_keep_prob = config.lstm_dropout_keep_prob
		lstm_layer_num = config.lstm_layer_num
		lstm_hidden_size = config.lstm_hidden_size
		is_training = config.is_training
		gan_hidden_dim = config.gan_hidden_dim
		alpha = config.alpha
		beta = config.beta
		learning_rate = config.learning_rate
		model = AnomalyNet(feature_index,
							FM_weight_dim,
							feature_item_num,
							batch_size,
							block_size,
							instance_dim,
							attention_dim,
							autoencoder_hidden_dim,
							lstm_dropout_keep_prob,
							lstm_layer_num,
							lstm_hidden_size,
							is_training,
							gan_hidden_dim,
							alpha,
							beta,
							learning_rate)
		saver = tf.train.Saver(max_to_keep=10)#saver for checkpoints, add var_list because of batching training
		
		init = tf.global_variables_initializer()  
		sess.run(init)
			 
		for epoch in range(config.epoch):
			# training
			for i in range(len(training_data)):
				curr_batch = training_data[i]	
				feed_dict = {model.data: curr_batch}
				result = sess.run((model.D_train,model.G_train,model.test1,model.test2),feed_dict=feed_dict)
				instance_loss_threshold = np.mean(result[2])
				block_loss_threshold = np.mean(result[3])
				print("current epoch %d, in batch %d, instance average loss %.4f, block average loss %.4f"%(epoch,i,instance_loss_threshold,block_loss_threshold),result[2].shape,result[3].shape)
			
			model_path = "saved_model/epoch_%s.ckpt" % (epoch)
			saver.save(sess, model_path) 
			'''
			#####
			testing 
			#####
			'''
			instance_loss_threshold = instance_loss_threshold * config.threshold_scale
			block_loss_threshold = block_loss_threshold * config.threshold_scale
			instance_pred = []
			block_pred = []
			for i in range(len(testing_data)):
				curr_batch = testing_data[i]	
				feed_dict = {model.data: curr_batch}
				instance_loss,block_loss = sess.run((model.instance_loss,model.block_loss),feed_dict=feed_dict)
				for j in range(len(instance_loss)): 
					if np.mean(instance_loss[j]) < instance_loss_threshold:
						instance_pred.append(1)
					else:
						instance_pred.append(0)

				for j in range(len(block_loss)): 
					if np.mean(block_loss[j]) < block_loss_threshold:
						block_pred.append(1)
					else:
						block_pred.append(0)
				
				# print("every batch instance/block:",instance_loss.shape,block_loss.shape)
			
			testing_block_num = testing_data_num // config.block_size
			block_true = []	
			#block anomaly detection from instance level. which means instances can also judge 		
			block_pred_from_instance = []
			for i in range(testing_block_num):
				pred_sum = np.sum(instance_pred[i*config.block_size:(i+1)*config.block_size])
				true_sum = np.sum(testing_label[i*config.block_size:(i+1)*config.block_size])
				
				#instance prediction judges block. If instance prediction very ensure
				# the block is anomaly or nomal, we use its result. Otherwise use our prediction result
				if pred_sum > config.block_size*config.instance_confidence:
					block_pred_from_instance.append(1)
				elif pred_sum < config.block_size*(1-config.instance_confidence):
					block_pred_from_instance.append(0)
				else:
					block_pred_from_instance.append(-1)
				# generate ground truth	
				if true_sum < config.block_size*config.block_ratio:
					block_true.append(0)
				else:
					block_true.append(1)
			
			block_pred_mixure = []
			for i in range(len(block_pred)):
				if block_pred_from_instance[i] == 0:
					block_pred_mixure.append(0)
				elif block_pred_from_instance[i] == 1:
					block_pred_mixure.append(1)
				else:
					block_pred_mixure.append(block_pred[i])
			#instance level evaluation
			print("testing_block_num",testing_data_num,testing_block_num,config.block_size)
			instance_eval = eval(testing_label,instance_pred)	
			#block level evaluation 
			block_eval = eval(block_true,block_pred_mixure)
			# print(block_true,block_pred)
			print("instance level evaluation: ",instance_eval)
			print("block level evaluation: ",block_eval)
if __name__ == '__main__':
	args = parse_args()
	run(args)








