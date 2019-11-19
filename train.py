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
import newmetrics

def parse_args():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
	parser.add_argument('--input', default='/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy/format.txt', help='data path')
	parser.add_argument('--instance-output', default='/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy/output/instance_output.txt', help='data path')
	parser.add_argument('--block-output', default='/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy/output/block_output.txt', help='data path')
	parser.add_argument('--FM-weight-dim', type=int, default=12, help='FM_weight_dim')
	parser.add_argument('--batch-size', type=int, default=4, help='batch size')
	parser.add_argument('--block-size', type=int, default=2, help='block size')
	parser.add_argument('--attention-dim', type=int, default=12, help='hidden dimension number')
	parser.add_argument('--autoencoder-hidden-dim', type=int, default=12, help='autoencoder dimension number')
	parser.add_argument('--lstm_dropout_keep_prob', type=float, default=1.0, help='drop out rate')
	parser.add_argument('--lstm-layer-num', type=int, default=1, help='LSTM layer number')
	parser.add_argument('--lstm-hidden-size', type=int, default=12, help='LSTM hidden layer dim')
	parser.add_argument('--gan-hidden-dim', type=int, default=12, help='LSTM hidden layer dim')
	parser.add_argument('--epoch', type=int, default=1, help='epoch number')
	parser.add_argument('--alpha', type=float, default=1.0, help='alpha for loss')
	parser.add_argument('--beta', type=float, default=1.0, help='beta for loss')
	parser.add_argument('--noise', type=float, default=0.0, help='noise for autoencoder')
	parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate')
	parser.add_argument('--block-ratio', type=float, default=0.5, help='block ratio')
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
	# feature_item_num = np.sum(dataset.feature_item_num)
	feature_item_num = dataset.feature_item_num # number of unique item ids in dataset instance
	instance_num = len(data)
	#for training
	training_data = data[:instance_num-2*anomaly_num]
	training_data = ld.get_shaped_data(training_data,config.batch_size,config.block_size,len(data[0]))
	print("----------finish shaping training data!-----------")
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
	print("----------finish shaping testing data!-----------")
	testing_data_num = len(testing_label) - len(testing_label)%(config.block_size*config.batch_size)
	testing_label = testing_label[:testing_data_num] # testing data instance level ground truth

	print("training data",training_data.shape,instance_dim)
	print("testing data",testing_data.shape,testing_data_num,testing_data[0].shape)
	print("anomaly_num",anomaly_num)
	print("number of normal data in testing data:",np.sum(testing_label),len(testing_label))
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
		noise = config.noise
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
							noise,
							learning_rate)
		saver = tf.train.Saver(max_to_keep=10)#saver for checkpoints, add var_list because of batching training
		
		init = tf.global_variables_initializer()  
		sess.run(init)
		
		flag = 0
		for epoch in range(config.epoch):
			# training
			for i in range(len(training_data)):
				flag = flag + 1
				pointer = flag % 100
				curr_batch = training_data[i]	
				feed_dict = {model.data: curr_batch}
				if pointer < 50:
					result = sess.run((model.G_train),feed_dict=feed_dict)
				else:
					result = sess.run((model.D_train),feed_dict=feed_dict)
				# result = sess.run((model.G_train,model.D_train),feed_dict=feed_dict)	
				if i % 50 == 0:
					result = sess.run((model.generator_loss,model.discriminator_loss),feed_dict=feed_dict)
					print("current epoch %d, in batch %d, current flag is %d, generator average loss %.4f, discriminator average loss %.4f"%(epoch,i,pointer,result[0],result[1]))
					# result = sess.run((model.test1,model.test2,model.test3,model.test4),feed_dict=feed_dict)
					# print(result[0],result[0].shape,result[1],result[1].shape)#,result[2][0:10],result[2].shape,result[3],result[3].shape)
			
			# model_path = "saved_model/epoch_%s.ckpt" % (epoch)
			# saver.save(sess, model_path) 
			# '''
			# #####
			# testing 
			# #####
			# '''

			#instance output 
			instance_loss_list = []
			block_loss_list = []
			for i in range(len(testing_data)):
				curr_batch = testing_data[i]	
				feed_dict = {model.data: curr_batch}
				instance_loss,block_loss = sess.run((model.instance_total_loss,model.block_total_loss),feed_dict=feed_dict)
				for i in range(len(instance_loss)):
					instance_loss_list.append(instance_loss[i])

				for i in range(len(block_loss)):
					block_loss_list.append(block_loss[i])

			bw = open(args.instance_output+'_%d'%(epoch), 'w')#by dingfu
			bw.write("true pred\n")
			for i in range(len(instance_loss_list)):
				bw.write(str(testing_label[i])+ " "+str(instance_loss_list[i])+"\n") 
			bw.close()
					

			#block output 
			testing_block_num = testing_data_num // config.block_size
			block_true = []	
			for i in range(testing_block_num):
				true_sum = np.sum(testing_label[i*config.block_size:(i+1)*config.block_size])
				
				# generate ground truth	
				if true_sum < config.block_size*config.block_ratio:
					block_true.append(0)
				else:
					block_true.append(1)
			
			bw = open(args.block_output+'_%d'%(epoch), 'w')#by dingfu
			bw.write("true pred\n")
			for i in range(testing_block_num):
				bw.write(str(block_true[i])+ " "+str(block_loss_list[i])+"\n")
			bw.close()

			# print(true_block,pred_block)
			instance_auc,_,_,_ = newmetrics.roc(testing_label,instance_loss_list,pos_label=0,output_path=args.instance_output+'_%d'%(epoch))#by dingfu
			block_auc,_,_,_ = newmetrics.roc(block_true,block_loss_list,pos_label=0,output_path=args.block_output+'_%d'%(epoch))#by dingfu
			#print("instance level evaluation: ",instance_eval)
			print('epoch:',epoch," instance level auc: ",instance_auc)
			#print("block level evaluation: ",block_eval)				
			print('epoch:',epoch," block level auc: ",block_auc)	

			
if __name__ == '__main__':
	args = parse_args()
	run(args)








