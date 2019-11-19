#!/usr/bin/python
# -*- coding:  utf-8 -*- 

'''
Configuration: parameter settings used in the whole model
Model: graph structure 
@author: 
Zheng Gao (gao27@indiana.edu) 
'''
from __future__ import print_function
import numpy as np
import tensorflow as tf 

class Config(object):
	FM_weight_dim = 5 
	block_size = 2
	batch_size = 2
	attention_dim = 5
	autoencoder_hidden_dim = 10
	lstm_dropout_keep_prob = 1.0
	lstm_layer_num = 1
	lstm_hidden_size = 6
	is_training = True
	gan_hidden_dim = 5


	max_epochs = 50
	dropout_keep_prob = 0.8
	lr_decay = 0.8
	source_embedding_size = 100
	target_embedding_size = 100
	source_hidden_size = 100
	target_hidden_size = 100
	attention_layer_size = 100
	num_layers = 2
	beam_width = 10
	max_grad_norm = 5
	learning_rate = 0.001

class AnomalyNet(object): 

	def __init__(self,feature_index,FM_weight_dim,feature_id_num,batch_size,
		block_size,instance_dim,attention_dim,autoencoder_hidden_dim,lstm_dropout_keep_prob,
		lstm_layer_num,lstm_hidden_size,is_training,gan_hidden_dim):
		self.feature_index = feature_index # the beginning index of each feature in the concatenate feature vector, starts with 0 and ends with the concatenate feature vector length
		self.FM_weight_dim = FM_weight_dim # the dimension of the weight embeddings
		self.feature_id_num = feature_id_num # the dimension (number of id) in each feature, stored in a list
		self.batch_size = batch_size 
		self.block_size = block_size
		self.instance_dim = instance_dim # the dimension of the concatenated feature, which is the input
		self.attention_dim = attention_dim #attention vector dimension
		self.autoencoder_hidden_dim = autoencoder_hidden_dim # autoencoder hidden layer dimension
		self.lstm_dropout_keep_prob = lstm_dropout_keep_prob #LSTM dropout keep probability
		self.lstm_layer_num = lstm_layer_num #number of layers in lstm
		self.lstm_hidden_size = lstm_hidden_size # block vector dimension
		self.is_training = is_training
		self.gan_hidden_dim = gan_hidden_dim

		#Input data, 3-D tensor [batch_size,block_size,intance_dim]
		self.data = tf.placeholder(tf.int32, shape=[None,self.block_size,self.instance_dim], name="train_features") 
		
		#FM feature vector calculation. 4-D tensor. [feature_num,batch_size,block_size,2*FM_weight_dim]
		self.feature_vectors = self.FM()

		#Attention mechanism to calculate instance vector. 3-D tensor. [batch_size,block_size,2*FM_weight_dim]
		self.instance_vectors_real = self.Attention()

		#auto-encoder as GAN generator 
		self.instance_vectors_fake = self.Autoencoder() #3-D tensor. [batch_size,block_size,2*FM_weight_dim]
		
		#LSTM to generate real/fake block vector
		self.block_vector_real,self.block_vector_fake = self.LSTM() #2-D tensor [batch_size,lstm_hidden_size]

		#GAN to transform instance / block vector
		self.GAN_discriminator() 
		#GAN generator loss
		I_generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_I_fake, labels=tf.ones_like(self.logit_I_fake))) 
		B_generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_B_fake, labels=tf.ones_like(self.logit_B_fake))) 
		self.generator_loss = I_generator_loss + B_generator_loss
		#GAN discriminator loss
		
		I_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_I_real, labels=tf.ones_like(self.logit_I_real)))
		I_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_I_fake, labels=tf.zeros_like(self.logit_I_fake)))  
		B_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_B_real, labels=tf.ones_like(self.logit_B_real)))
		B_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_B_fake, labels=tf.zeros_like(self.logit_B_fake)))  
		self.discriminator_loss= I_real_loss+I_fake_loss+B_real_loss+B_fake_loss
		

		self.D_train = tf.train.AdamOptimizer().minimize(self.discriminator_loss)
		self.G_train = tf.train.AdamOptimizer().minimize(self.generator_loss)

	def FM(self):
		# initialize all weight emebddings 
		self.feature_weight_embedding_list = [] 
		with tf.variable_scope("FM"): 
			for i in range(len(self.feature_id_num)):
				first_order_weight = tf.get_variable('weight_1_embeddings_feature_{}'.format(i),initializer= tf.ones([self.feature_id_num[i],self.FM_weight_dim]), dtype=tf.float32) 
				second_order_weight = tf.get_variable('weight_2_embeddings_feature_{}'.format(i),initializer= tf.ones([self.feature_id_num[i],self.FM_weight_dim]), dtype=tf.float32) 
				self.feature_weight_embedding_list.append((first_order_weight,second_order_weight))# form a tuple and append to the list

		# segment the concatenated feature into several individual features
		self.tensor_list = []
		for i in range(len(self.feature_index)-1):
			temp = tf.slice(self.data,[0,0,self.feature_index[i]],[tf.shape(self.data)[0],tf.shape(self.data)[1],self.feature_index[i+1]-self.feature_index[i]])
			self.tensor_list.append(temp)
		
		#calculate FM feature level representation, in each feature
		feature_vectors = []
		for i in range(len(self.tensor_list)):
			sliced_feature = self.tensor_list[i]  #[batch_size,block_size,feature_size]
			related_weight_first_order = self.feature_weight_embedding_list[i][0]
			related_weight_second_order = self.feature_weight_embedding_list[i][1]

			'''
			calculate first order vector: sum all id vectors together, and substract all id=0 vectors 
			'''

			first_order_vec = tf.nn.embedding_lookup(related_weight_first_order,sliced_feature) # 4-D
			sliced_feature_first = tf.clip_by_value(sliced_feature, 0, 1) # convert all eligible ids to 1, keeps the dummy id 0 as 0
			sliced_feature_first = tf.expand_dims(sliced_feature_first, -1) # expend one more dimension in the last dimension 

			first_order_vec = tf.multiply(first_order_vec,tf.cast(sliced_feature_first,tf.float32)) # dot product to get eligible id vectors. dummy id 0 has a vector 0
			first_order_vec = tf.reduce_sum(first_order_vec,axis=2,name='fm_first') #3-D, sum on the third dimension. sum up all id vectors in the same instance
			
			# self.first_order_vec = first_order_vec

			'''
			calculate second order vector:
			'''
			second_order_vec = tf.nn.embedding_lookup(related_weight_second_order,sliced_feature) # 4-D
			sliced_feature_second = tf.clip_by_value(sliced_feature, 0, 1) # convert all eligible ids to 1, keeps the dummy id 0 as 0
			sliced_feature_second = tf.expand_dims(sliced_feature_second, -1) # expend one more dimension in the last dimension 

			second_order_vec = tf.multiply(second_order_vec,tf.cast(sliced_feature_second,tf.float32)) # dot product to get eligible id vectors. dummy id 0 has a vector 0
			
			#sum square
			second_order_vec_sum = tf.reduce_sum(second_order_vec,axis=2) #3-D, sum on the third dimension. sum up all id vectors in the same instance
			second_order_sum_square = tf.square(second_order_vec_sum)# (a+b)^2

			#square sum
			second_order_vec_square = tf.square(second_order_vec)
			second_order_square_sum = tf.reduce_sum(second_order_vec_square,axis=2)# a^2+b^2

			#second order vector
			second_order_vec = 0.5 * tf.subtract(second_order_sum_square,second_order_square_sum,name='fm_second')
			# self.second_order_vec = second_order_vec
			

			# concatenate two first order and second order feature together 
			current_feature_vec = tf.concat([first_order_vec,second_order_vec],-1)
			# self.current_feature_vec = current_feature_vec
			feature_vectors.append(current_feature_vec)
			# break
		feature_vectors = tf.stack(feature_vectors,axis = 0) # stack all feature vectors together to form a uniformed tensor
		return feature_vectors

	def Attention(self):
		with tf.variable_scope("attention"): 
			self.att_v = tf.get_variable('vector',initializer= tf.ones([1,self.attention_dim]), dtype=tf.float32) 
			self.att_w = tf.get_variable('weight',initializer= tf.ones([2*self.FM_weight_dim,self.attention_dim]), dtype=tf.float32) #2-D [feature_vector_dim,attention_dim]
			self.att_b = tf.get_variable('bias',initializer= tf.ones([1,self.attention_dim]), dtype=tf.float32) 
	
		data_transpose = tf.transpose(self.feature_vectors,[1,2,0,3]) # turn to batch_size, block_size, feature_num, feature_vector_dim
		weight = tf.tile(tf.reshape(self.att_w,[1,1,tf.shape(self.att_w)[0],tf.shape(self.att_w)[1]]),[tf.shape(data_transpose)[0],tf.shape(data_transpose)[1],1,1]) 
		bias = tf.tile(tf.reshape(self.att_b,[1,1,tf.shape(self.att_b)[0],tf.shape(self.att_b)[1]]),[tf.shape(data_transpose)[0],tf.shape(data_transpose)[1],len(self.feature_index)-1,1])#feature number
		vector = tf.tile(tf.reshape(self.att_v,[1,1,tf.shape(self.att_v)[0],tf.shape(self.att_v)[1]]),[tf.shape(data_transpose)[0],tf.shape(data_transpose)[1],1,1])
		u= tf.tanh(tf.matmul(data_transpose,weight)+bias)
		vu = tf.matmul(u, tf.transpose(vector, [0,1,3,2]))  
		exp = tf.exp(vu)
		exp_sum = tf.reduce_sum(exp, 2) 
		shape = tf.shape(exp_sum)
		exp_sum = tf.reshape(exp_sum, [shape[0],shape[1],shape[2],-1]) 
		alphas = exp / exp_sum

		instance_vector = tf.reduce_sum(data_transpose*alphas,2)
		return instance_vector

	def Autoencoder(self):
		with tf.variable_scope("autoencoder"):
			self.w_enc = tf.get_variable('weight_encoder',initializer= tf.ones([2*self.FM_weight_dim,self.autoencoder_hidden_dim]), dtype=tf.float32) 
			self.b_enc = tf.get_variable('bias_encoder',initializer= tf.ones([1,self.autoencoder_hidden_dim]), dtype=tf.float32) 

			self.w_dec = tf.get_variable('weight_decoder',initializer= tf.ones([self.autoencoder_hidden_dim,2*self.FM_weight_dim]), dtype=tf.float32) 
			self.b_dec = tf.get_variable('bias_decoder',initializer= tf.ones([1,2*self.FM_weight_dim]), dtype=tf.float32) 
			self.hidden = self.encoder()
			instance_vectors_fake = self.decoder()
			return instance_vectors_fake
	def encoder(self):
		#input to hidden layer
		# input are 3-D: batch_size, block_size, instance_dim
		# convert to 3-D:block_size, batch_size, instance_dim
		# after transformation: block_size, batch_size, autoencoder_hidden_dim
		data = tf.transpose(self.instance_vectors_real,[1,0,2])
		w_enc = tf.reshape(self.w_enc,[-1,tf.shape(self.w_enc)[0],tf.shape(self.w_enc)[1]])
		w_enc = tf.tile(w_enc,[tf.shape(data)[0],1,1])
		b_enc = tf.reshape(self.b_enc,[-1,tf.shape(self.b_enc)[0],tf.shape(self.b_enc)[1]])
		b_enc = tf.tile(b_enc,[tf.shape(data)[0],tf.shape(data)[1],1])
		hidden  = tf.matmul(data,w_enc)+b_enc
		hidden = tf.nn.sigmoid(hidden)
		# self.hidden = b_enc
		return hidden 

	def decoder(self):
		#hidden layer to output 
		#hidden layer 3-D:block_size, batch_size, autoencoder_hidden_dim
		#output 3-D: batch_size, block_size, instance_dim
		w_dec = tf.reshape(self.w_dec,[-1,tf.shape(self.w_dec)[0],tf.shape(self.w_dec)[1]])
		w_dec = tf.tile(w_dec,[tf.shape(self.hidden)[0],1,1])
		b_dec = tf.reshape(self.b_dec,[-1,tf.shape(self.b_dec)[0],tf.shape(self.b_dec)[1]])
		b_dec = tf.tile(b_dec,[tf.shape(self.hidden)[0],tf.shape(self.hidden)[1],1])

		instance_vectors_fake = tf.matmul(self.hidden,w_dec)+b_dec
		instance_vectors_fake = tf.nn.sigmoid(instance_vectors_fake)
		instance_vectors_fake = tf.transpose(instance_vectors_fake,[1,0,2]) #get back to 3-D: batch_size, block_size, autoencoder_hidden_dim
		return instance_vectors_fake
	
	def LSTM(self):
		#for real data
		lstm_cells_real = [tf.nn.rnn_cell.DropoutWrapper(
					tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_size),
					output_keep_prob = self.lstm_dropout_keep_prob)
					for _ in range(self.lstm_layer_num)]
		cell_real = tf.nn.rnn_cell.MultiRNNCell(lstm_cells_real)

		self.initial_state_real = cell_real.zero_state(self.batch_size,tf.float32)

		if self.is_training:
			self.instance_vectors_real = tf.nn.dropout(self.instance_vectors_real,self.lstm_dropout_keep_prob)
		state = self.initial_state_real
		with tf.variable_scope("LSTM_real"): 
			for time_step in range(self.block_size):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				cell_output_real,state = cell_real(self.instance_vectors_real[:,time_step,:],state) # the last step output is regarded as the block vector
		
		#for fake data
		lstm_cells_fake = [tf.nn.rnn_cell.DropoutWrapper(
					tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_size),
					output_keep_prob = self.lstm_dropout_keep_prob)
					for _ in range(self.lstm_layer_num)]
		cell_fake = tf.nn.rnn_cell.MultiRNNCell(lstm_cells_fake)

		self.initial_state_fake = cell_fake.zero_state(self.batch_size,tf.float32)

		if self.is_training:
			self.instance_vectors_fake = tf.nn.dropout(self.instance_vectors_fake,self.lstm_dropout_keep_prob)
		
		'''
		Nocice!!! Mistery, I need to provid a real number for the last dimension, so I need to reshape 
		
		'''
		self.instance_vectors_fake = tf.reshape(self.instance_vectors_fake,[tf.shape(self.instance_vectors_fake)[0],tf.shape(self.instance_vectors_fake)[1],2*self.FM_weight_dim])
		state = self.initial_state_fake
		with tf.variable_scope("LSTM_fake"): 
			for time_step in range(self.block_size):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				cell_output_fake,state = cell_fake(self.instance_vectors_fake[:,time_step,:],state) # the last step output is regarded as the block vector
		
		return cell_output_real,cell_output_real
	def GAN_discriminator(self):
		with tf.variable_scope("instance"): 
			self.w_instance_hidden = tf.get_variable('w1',initializer= tf.ones([2*self.FM_weight_dim,self.gan_hidden_dim]), dtype=tf.float32)  
			self.b_instance_hidden = tf.get_variable('b1',initializer= tf.ones([1,self.gan_hidden_dim]), dtype=tf.float32) 

			self.w_instance_output = tf.get_variable('w2',initializer= tf.ones([self.gan_hidden_dim,1]), dtype=tf.float32) 
			self.b_instance_output = tf.get_variable('b2',initializer= tf.ones([1,1]), dtype=tf.float32) 

		with tf.variable_scope("block"): 
			self.w_block_hidden = tf.get_variable('w1',initializer= tf.ones([self.lstm_hidden_size,self.gan_hidden_dim]), dtype=tf.float32) 
			self.b_block_hidden = tf.get_variable('b1',initializer= tf.ones([1,self.gan_hidden_dim]), dtype=tf.float32) 

			self.w_block_output = tf.get_variable('w2',initializer= tf.ones([self.gan_hidden_dim,1]), dtype=tf.float32) 
			self.b_block_output = tf.get_variable('b2',initializer= tf.ones([1,1]), dtype=tf.float32) 

		self.logit_I_real = self.discriminator_instance(self.instance_vectors_real)
		self.logit_I_fake = self.discriminator_instance(self.instance_vectors_fake)
		self.logit_B_real = self.discriminator_block(self.block_vector_real)
		self.logit_B_fake = self.discriminator_block(self.block_vector_fake)

		

	def discriminator_instance(self,data):
		instance = tf.reshape(data,[-1,2*self.FM_weight_dim])
		hidden_layer = tf.nn.relu(tf.matmul(instance, self.w_instance_hidden) + self.b_instance_hidden)
		output = tf.matmul(hidden_layer,self.w_instance_output)+self.b_instance_output
		return output

	def discriminator_block(self,data):
		hidden_layer = tf.nn.relu(tf.matmul(data, self.w_block_hidden) + self.b_block_hidden)
		output = tf.matmul(hidden_layer,self.w_block_output)+self.b_block_output
		return output






