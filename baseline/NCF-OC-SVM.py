#!/usr/bin/python
# -*- coding:  utf-8 -*-
'''
Use NCF to get a dense vector,
and use One class SVM for anomaly detection
convolutional autoencoder: https://github.com/raghavchalapathy/oc-nn/blob/master/models/CAE_OCSVM_models.py
'''
from __future__ import print_function
import numpy as np 
from sklearn import metrics
from sklearn import svm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import loadData as ld 
import random

def parse_args():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
	parser.add_argument('--input', default='/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy/format.txt', help='data path')
	parser.add_argument('--instance-output', default='/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy/instance_output.txt', help='data path')
	parser.add_argument('--block-output', default='/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/toy/block_output.txt', help='data path')
	parser.add_argument('--block-size', type=int, default=2, help='block size')
	parser.add_argument('--block-ratio', type=float, default=0.9, help='gamma for loss')
	parser.add_argument('--categorical', dest='categorical', action='store_true')
	parser.add_argument('--no-categorical', dest='categorical', action='store_false')
	parser.set_defaults(categorical=True)
	args = parser.parse_args()
	return args
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

def oneclassSVM(args):
	#load data
	path = args.input
	if args.categorical:
		training_data,training_label,testing_data,testing_label = ld.load_categorical(path)
	else:
		training_data,training_label,testing_data,testing_label = ld.load_numeric(path) 
	#shuffle testing data,to ensure testing data and label are shuffled in the same way
	randnum = random.randint(0,100)
	random.seed(randnum)
	random.shuffle(testing_data)
	random.seed(randnum)
	random.shuffle(testing_label)
	
	clf = svm.OneClassSVM(kernel="rbf")
	clf.fit(training_data)
	label_pred = clf.predict(testing_data)  
	individual_pred = []
	for i in range(len(label_pred)):
		if label_pred[i] == -1:
			individual_pred.append(0)
		else:
			individual_pred.append(label_pred[i])

	instance_eval = eval(testing_label,individual_pred)		
			
	score = clf.score_samples(testing_data)
	print("data",testing_label,label_pred,score)
	#write instance loss to file
	bw = open(args.instance_output, 'w')
	bw.write("true pred\n")
	for i in range(len(score)):
		bw.write(str(testing_label[i])+ " "+str(score[i])+"\n")
	bw.close()
	#block level evaluation
	pred_block = []
	true_block = []
	testing_block_num = len(testing_data) // args.block_size
	for i in range(testing_block_num):
		pred_sum = np.sum(individual_pred[i*args.block_size:(i+1)*args.block_size])
		true_sum = np.sum(testing_label[i*args.block_size:(i+1)*args.block_size])
		if pred_sum < args.block_size*args.block_ratio:
			pred_block.append(0)
		else:
			pred_block.append(1)
		if true_sum < args.block_size*args.block_ratio:
			true_block.append(0)
		else:
			true_block.append(1)
	block_eval = eval(true_block,pred_block)

	#write block loss to file
	bw = open(args.block_output, 'w')
	bw.write("true pred\n")
	for i in range(testing_block_num):
		true_block = "1"
		pred_block = np.mean(score[i*args.block_size:(i+1)*args.block_size])
		true_sum = np.sum(testing_label[i*args.block_size:(i+1)*args.block_size])
		if true_sum < args.block_size*args.block_ratio:
			true_block = "0"
		bw.write(true_block+ " "+str(pred_block)+"\n")
	bw.close()
	print("instance level evaluation: ",instance_eval)
	print("block level evaluation: ",block_eval)


if __name__ == '__main__':
	args = parse_args()
	oneclassSVM(args)