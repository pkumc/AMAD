#!/usr/bin/python
# -*- coding:  utf-8 -*- 
from __future__ import print_function
import numpy as np
import tensorflow as tf
# from backups.model import AnomalyNetConfig
import math
from matplotlib import pyplot as plt
import random
from os import listdir
from os.path import isfile, join 
from scipy import io
from random import shuffle

def test_tf_data(batch_size):# a demo code to use tensorflow dataset module
	X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
	y = np.ones(5)
	label = np.zeros(5)
	dataset = tf.data.Dataset.from_tensor_slices(X)
	dataset = dataset.shuffle(5) #maintain a shuffle, refer to https://juejin.im/post/5b855d016fb9a01a1a27d035
	dataset = dataset.repeat()# endless repeat the dataset
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_one_shot_iterator()
	one_element = iterator.get_next()

	with tf.Session() as sess:
		for i in range(5):
			print("this is the %d th run" % (i))
			print(sess.run(one_element))
	# return dataset

def synthetic_data(instance_num,instance_dim,cycle,feature_num): 
	data = []
	for f in range(feature_num):
		feature = [] 
		sequence = np.random.choice(instance_dim, cycle, replace=True)
		cycle_num = int(instance_num/cycle)
		cycle_remain = int(instance_num%cycle)
		for i in range(cycle_num):
			for j in range(len(sequence)):
				rand = random.random()
				if rand < 0.1:
					r = np.random.choice(instance_dim, 1)[0]
					feature.append(r)
				else:
					feature.append(sequence[j])
			# feature.extend(sequence)
		for i in range(cycle_remain):
			rand = random.random()
			if rand < 0.1:
				r = np.random.choice(instance_dim, 1)[0]
				feature.append(r)
			else:
				feature.append(sequence[j]) 
		# plt.scatter([x for x in range(len(feature))],feature)
		data.append(feature)
		# break
	# plt.show()
	# speed = [0.05,0.025,0.01,0.005]
	# for s in speed: 
		# rand = np.random.normal(0, 0.2, size)
		# rand = np.zeros(size)
		# feature = [x%(dim) for x in range(size)]
		# feature = [(math.sin(s*x*math.pi*2)+1)*dim/2 for x in range(size)] 
		# feature = [min(abs(math.floor(feature[x]+rand[x])),dim-1) for x in range(size)]
		# data.append(feature)
		# print(feature) 
		# print([feature2[x]-math.floor(feature[x]) for x in range(size)])
		# plt.plot(feature)
		# print([math.floor(x) for x in rand])
		# plt.plot([math.floor(x) for x in rand])
		# plt.plot(f4)
		# break
	# plt.show()
	# return [f1,f2,f3,f4]
	# idx = [x for x in range(size)]
	# f1 = []
	# for x in range(size):
	# 	if x%cycle < dim:
	# 		f1.append(x%cycle)
	# 	else:
	# 		f1.append(x%cycle-dim) 
	# plt.scatter(idx,f1)
	# plt.show()
	return data

def write_data_to_file(data,file):
	bw = open(file, 'w')
	num = len(data[0])
	for i in range(num):
		output = ""
		for j in range(len(data)):
			output = output + str(data[j][i]) + ":1 "
		output = output.rstrip() + "\n"
		bw.write(output)
	bw.close()

# def generate_test_data_format():#generate particular format of input data to test which one is the best input format

def generate_sythetic_data(output_file,id_range,instance_N):
	instance_l = [] 
	instance_dim = 3
	start_index = 1
	for i in range(instance_N): 
		output = "-1 " 
		instance_index = start_index
		for j in range(instance_dim):
			rand = random.uniform(0, 1)
			if rand > 0.8:
				instance_index = (instance_index + 1)% id_range
			else:
				instance_index = (instance_index - 1)% id_range
			# if instance_index == 0:
			# 	instance_index = 10
			output = output + str(instance_index+10*j+1) + "," 
		instance_l.append(output[:-1])

		rand = random.uniform(0, 1)
		if rand > 0.8:
			start_index = (start_index + 1)% id_range
		else:
			start_index = (start_index - 1)% id_range
	
	# print(instance_l)	
	result = []
	result.extend(instance_l[0:5000])	
	noise = []
	multiply = [1,11,21]
	for i in range(100):
		 temp =  list(np.random.choice(10, instance_dim))
		 for j in range(instance_dim):
		 	temp[j] = temp[j]+multiply[j]
		 noise.append("0 "+",".join(str(x) for x in temp))
	# print(noise)
	for i in range(5,10):
		result.extend(noise)
		for j in range(i*1000,i*1000+100):
			instance_l[j] = instance_l[j].replace("-","")
			result.append(instance_l[j]) 
		result.extend(instance_l[i*1000+100:(i+1)*1000])

	bw = open(output_file, 'w')
	for i in range(len(result)):
		# print(result[i],type(result[i]))
		bw.write(result[i]+'\n')
	bw.close()

def convert_data(path):
	onlyfiles = [path+f for f in listdir(path) if isfile(join(path, f))] 
	print(onlyfiles)
	for file in onlyfiles:
		bw = open(file+"_new", 'w')
		normal = []
		anomaly = []
		with open(file) as f:	
			for line in f:
				result = line.rstrip().split(" ")
				label = int(result[0]) 
				if label == 0:
					anomaly.append(line)
				else:
					normal.append(line.replace("-",""))
		for i in range(len(normal)):
			bw.write(normal[i])
		for i in range(len(anomaly)):
			bw.write(anomaly[i])
		bw.close()

def load_mat_data(file):
	mat = io.loadmat(file)
	print(mat.keys()) 
	print(mat.values()) 

def convert_public_data(input_file,output_file):
	bw = open(output_file, 'w')
	with open(input_file) as f:
		for line in f:
				result = line.rstrip().split(" ")
				output = result[0]+ " " + ",".join(result[i] for i in range(1,len(result))) +"\n"
				bw.write(output)
	bw.close()

def shuffle_public_data(input_file,output_file):
	# bw = open(output_file, 'w')
	with open(input_file) as f:
		for line in f:
				result = line.rstrip().split(" ")
				if result[0] == "1":
					bw.write(line)
				else:
					temp =[]
					print(len(result))
					for i in range(1,len(result)):
						temp.append(result[i])
						print(len(result[i].split(",")))
					# shuffle(temp)
					# output = result[0]+ " " + " ".join(x for x in temp) +"\n"
					# bw.write(output)
	# bw.close()

def merge_file(input_file1,input_file2,output_file):
	bw = open(output_file, 'w')
	with open(input_file1) as f:
		for line in f:
			bw.write(line)
	with open(input_file2) as f:
		for line in f:
			bw.write(line)
	bw.close()
if __name__ == '__main__':
	# test_tf_data(AnomalyNetConfig.batch_size)
	# data = synthetic_data(500,10,50,4) # dimension should smaller than cycle. Otherwise we can't cover all numbers
	# write_data_to_file(data,"sample.txt")


	# generate_sythetic_data("/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/synthetic/synthetic.txt",10,10000)
	# path = "/Users/zhenggao/Desktop/alibaba/阿里妈妈/data/public/realTweetsData/"
	# # convert_data(path)
	# load_mat_data("/Users/zhenggao/Downloads/shuttle.mat")
	input_file1 = "/Users/zhenggao/Desktop/data_normal.csv"
	input_file2= "/Users/zhenggao/Desktop/data_anormal.txt"
	output_file = "/Users/zhenggao/Desktop/alimama.txt"
	merge_file(input_file1,input_file2,output_file)
	# convert_public_data(input_file,output_file)
	# shuffle_public_data(input_file,output_file)




	
