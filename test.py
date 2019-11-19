#!/usr/bin/python
# -*- coding:  utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn import metrics
#shape：数据形状，选填，默认为value的shape，设置时不得比value小，可以比value阶数、维度更高，超过部分按value提供最后一个数字填充
# feature = [1,2,3]
# constatnValue = tf.constant(feature)
# print(constatnValue.shape[0])
# x = tf.placeholder(dtype=tf.int32,name = 'x')
# b = tf.equal(tf.reduce_sum(x),tf.constant(1))
# print("hahahahh",b)
# # b = [0,1,1]
# # for i in b:
# # 	if i == 1:
# # 		c = x.slice()
# #创建一个会话
# session = tf.Session()
# print(session.run(constatnValue))
# print(session.run((x,b),feed_dict={x:feature}))
# #关闭会话
# session.close() 

# dataset = [[[1], [2], [3], [4]], 
# 			[[2], [4], [1]],
# 			[[5], [2], [7], [3], [8]],
# 			[[9], [7]]]
# feature_dim = 1
# num_samples = len(dataset) # 序列的个数。输出：4
# lengths = [len(s) for s in dataset] # 获取每个序列的长度。输出：[4, 3, 5, 2]
# max_length = max(lengths) # 最长序列的长度。输出：5
# padding_dataset = np.zeros([num_samples, max_length, feature_dim]) 
# padding_dataset2 = np.ones([num_samples, max_length, feature_dim])
# padding_dataset = padding_dataset - padding_dataset2
# # 生成一个全零array来存放padding后的数据集
# for idx, seq in enumerate(dataset): # 将序列放入array中（相当于padding成一样长度）
# 	padding_dataset[idx, :len(seq), :] = seq
# print(padding_dataset	)

# data = [[[[3,1,-1],[2],[4,3,1,2]],
# 		[[3,2,-1],[1],[2,3,-1,-1]],	
# 		[[1,-1,-1],[-1],[4,2,1,-1]]]] # one batch is a block
# data = np.asarray(data)
# print(data,data.shape)
# a = tf.constant([[[0,1],[3,3]],[[0,1],[0,0]]])
# weight = tf.constant([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]) 
# b = tf.nn.embedding_lookup(weight,a)

# a = tf.clip_by_value(a, 0, 1) 
# a = tf.expand_dims(a, -1)
# # a = tf.reshape(a,[tf.shape(a)[0],tf.shape(a)[1],tf.shape(a)[2],-1])
# # # a = tf.constant([-1,-1])
# c = tf.multiply(a,b)
# with tf.Session() as sess:
# 	result = sess.run((a,c))
# 	print(result)
# # temp1 = tf.zeros(tf.shape(a),dtype=tf.int32)
# # temp = tf.subtract(temp1,temp2)
# # bool_mask = tf.not_equal(a, temp)
# bool_mask = tf.count_nonzero(a,axis=2,dtype=tf.int32)
# # bool_mask = tf.reshape(bool_mask,[tf.shape(a)[0],tf.shape(a)[1],-1]) 
# temp = tf.multiply(tf.ones([tf.shape(a)[0],tf.shape(a)[1]],dtype=tf.int32),tf.shape(a)[-1])
# bool_mask = tf.subtract(temp,bool_mask)
# bool_mask = tf.reshape(bool_mask,[tf.shape(a)[0],tf.shape(a)[1],-1]) 


# zero_embedding = tf.nn.embedding_lookup(weight,0)
# zero_embedding = tf.reshape(zero_embedding,[1,1,-1]) 
# zero_embedding = tf.tile(zero_embedding,[tf.shape(bool_mask)[0],1,1])
# bool_mask = tf.matmul(bool_mask,zero_embedding)
# b = tf.nn.embedding_lookup(weight,a)
# c = tf.reduce_sum(b,axis=2)

# d = tf.subtract(c,bool_mask)
# session = tf.Session()
# result = session.run((d,c,bool_mask,zero_embedding)) 
# print(result[0],result[1],result[1].shape,result[2],result[2].shape,result[3],result[3].shape)
# session.close()

# a = tf.constant([[[1],[3]],[[5],[0]]])
# b = tf.constant([[[1,2],[3,4]],[[5,6],[0,0]]])
# c = tf.multiply(b,a)
# # zeros = tf.zeros_like(X)
# # index = tf.not_equal(X,zeros)
# # loc = tf.where(index,x=X,y=X)

# with tf.Session() as sess:
#     out = sess.run(c)
#     print(out)


				#1. for each feature, sum all ids vector together to form a overall vector representation
				# first_order_vec = tf.nn.embedding_lookup(related_weight_first_order,sliced_feature) # 4-D
				# first_order_vec = tf.reduce_sum(first_order_vec,axis=2) #3-D, sum on the third dimension

				# #2. for each feature, get the counts of  index 0, which are dummy index
				# # as index 0 in the weight embedding is a dummy index, we need to remove all 0 vectors out of the overall summed vector
				# index_zero = tf.count_nonzero(sliced_feature,axis=2,dtype=tf.float32) # 2-D count the number of non-zero ids in each instance
				# index_total = tf.multiply(tf.ones([tf.shape(sliced_feature)[0],tf.shape(sliced_feature)[1]],dtype=tf.float32),
				# 				tf.cast(tf.shape(sliced_feature)[-1],tf.float32)) 	#2-D count the number of ids in each intance
				# index_zero = tf.subtract(index_total,index_zero) #2-D number of zeros in each instance 
				# index_zero = tf.reshape(index_zero,[tf.shape(sliced_feature)[0],tf.shape(sliced_feature)[1],-1]) #3-D add one more dimension [batch_size,block_size,1]

				# # tile the zero embedding to match index_zero dimensions
				# zero_embedding = tf.nn.embedding_lookup(related_weight_first_order,0) #1-D
				# zero_embedding = tf.reshape(zero_embedding,[1,1,-1]) #3-D
				# zero_embedding = tf.tile(zero_embedding,[tf.shape(index_zero)[0],1,1])# tile the embedding to calculate the sum of zero embeddings
				# index_zero = tf.matmul(index_zero,zero_embedding) # 3-D calculate zero embedding sums for each instance

				# first_order_vec = tf.subtract(first_order_vec,index_zero)# 3-D [batch_size,block_size,weight_size]. Represent each instance in first order feature level


a = tf.constant(np.arange(1,17), shape=[4, 4],dtype=tf.float32)
b = tf.reshape(a,[2,2,2,2])
c = tf.constant(np.arange(1,9), shape=[4, 2],dtype=tf.float32)
d = tf.reshape(c,[2,2,1,2]) 
e = b/d
# b = tf.reshape(a,[2,2,3])
# # b = tf.constant(np.arange(1,7), shape=[1, 3, 2],dtype=tf.int32)
# # c = tf.matmul(a, b) 
# init = tf.global_variables_initializer()

with tf.Session() as sess:
#     sess.run(init)
 	result = sess.run((b,d,e))
 	print(result[0],result[1],result[2],result[0].shape,result[1].shape,result[2].shape)
#     print('a=', a.eval())
#     print('b=', b.eval())
#     # print('c=', c.eval())  		
# a = np.arange(1,13)
# b = np.reshape(a,[4,3])
# c = np.reshape(b,[2,2,3])

# print(a,b,c)

# def eval(truth,pred):
# 	evaluation_dict = {}
# 	acc = metrics.accuracy_score(truth,pred);evaluation_dict['acc']=acc
# 	precision = metrics.precision_score(truth,pred,pos_label=0);evaluation_dict['precision']=precision
# 	recall = metrics.recall_score(truth,pred,pos_label=0);evaluation_dict['recall']=recall
# 	f1_macro = metrics.f1_score(truth,pred, average='macro',pos_label=0);evaluation_dict['f1_macro']=f1_macro
# 	f1_micro = metrics.f1_score(truth,pred, average='micro',pos_label=0);evaluation_dict['f1_micro']=f1_micro
# 	mcc = metrics.matthews_corrcoef(truth,pred);evaluation_dict['mcc']=mcc
# 	# auc = metrics.roc_auc_score(truth,pred);evaluation_dict['auc']=auc
# 	return evaluation_dict

# if __name__ == '__main__':
# 	# pred = [0,0]
# 	# true = [0,1]
# 	# result = eval(true,pred)
	# print(result) 
