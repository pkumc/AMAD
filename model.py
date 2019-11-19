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
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

class Config(object):
	def __init__(self,args):
		self.FM_weight_dim = args.FM_weight_dim 
		self.block_size = args.block_size
		self.batch_size = args.batch_size
		self.attention_dim = args.attention_dim
		self.autoencoder_hidden_dim = args.autoencoder_hidden_dim
		self.lstm_dropout_keep_prob = args.lstm_dropout_keep_prob
		self.lstm_layer_num = args.lstm_layer_num
		self.lstm_hidden_size = args.lstm_hidden_size
		self.gan_hidden_dim = args.gan_hidden_dim
		self.epoch = args.epoch
		self.alpha = args.alpha
		self.beta = args.beta
		self.noise = args.noise
		self.learning_rate = args.learning_rate
		self.block_ratio = args.block_ratio
		self.learning_rate = args.learning_rate
		self.is_training = True
		self.seed = 20
class AnomalyNet(object): 

	def __init__(self,feature_index,FM_weight_dim,feature_item_num,batch_size,
		block_size,instance_dim,attention_dim,autoencoder_hidden_dim,lstm_dropout_keep_prob,
		lstm_layer_num,lstm_hidden_size,is_training,gan_hidden_dim,alpha,beta,noise,learning_rate):
		#in order to generate same random sequences 
		# tf.set_random_seed(1)

		self.feature_index = feature_index # the beginning index of each feature in the concatenate feature vector, starts with 0 and ends with the concatenate feature vector length
		self.FM_weight_dim = FM_weight_dim # the dimension of the weight embeddings
		self.feature_item_num = feature_item_num # the dimension (number of id) in each feature, stored in a list
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
		self.alpha = alpha
		self.beta = beta
		self.noise = noise
		self.learning_rate = learning_rate
		#batch normalization whether training or inference
		self.batch_norm = tf.placeholder_with_default(True, shape=(), name='batch_norm_is_training') 

		#Input data, 3-D tensor [batch_size,block_size,intance_dim]
		self.data = tf.placeholder(tf.int32, shape=[None,self.block_size,self.instance_dim], name="train_features") 
		
		
		# initialize all weight emebddings 
		
		with tf.variable_scope("item_attention"): 
			self.item_emb = tf.get_variable("item_embedding",initializer= tf.truncated_normal([self.feature_item_num,self.FM_weight_dim]), dtype=tf.float32) 
			self.item_att_v = tf.get_variable('vector',initializer= tf.truncated_normal([self.attention_dim,1]), dtype=tf.float32) 
			self.item_att_w = tf.get_variable('weight',initializer= tf.truncated_normal([self.FM_weight_dim,self.attention_dim]), dtype=tf.float32)
			self.item_att_b = tf.get_variable('bias',initializer= tf.truncated_normal([self.attention_dim]), dtype=tf.float32) 
		
		with tf.variable_scope("feature_attention"): 
			self.feature_att_v = tf.get_variable('vector',initializer= tf.truncated_normal([self.attention_dim,1]), dtype=tf.float32) 
			self.feature_att_w = tf.get_variable('weight',initializer= tf.truncated_normal([self.FM_weight_dim,self.attention_dim]), dtype=tf.float32) #2-D [feature_vector_dim,attention_dim]
			self.feature_att_b = tf.get_variable('bias',initializer= tf.truncated_normal([self.attention_dim]), dtype=tf.float32) 
		
		with tf.variable_scope("previous_block_attention"): 
			self.previous_block_att_v = tf.get_variable('previous_block',initializer= tf.truncated_normal([self.lstm_hidden_size]), dtype=tf.float32) 
			self.previous_block_att_w = tf.get_variable('weight',initializer= tf.truncated_normal([self.FM_weight_dim,self.lstm_hidden_size]), dtype=tf.float32) #2-D [feature_vector_dim,attention_dim]
			self.previous_block_att_b = tf.get_variable('bias',initializer= tf.truncated_normal([self.lstm_hidden_size]), dtype=tf.float32) 
		

		with tf.variable_scope("autoencoder"):
			self.w_enc = tf.get_variable('weight_encoder',initializer= tf.truncated_normal([(len(self.feature_index)-1)*self.FM_weight_dim,self.autoencoder_hidden_dim]), dtype=tf.float32) 
			self.b_enc = tf.get_variable('bias_encoder',initializer= tf.truncated_normal([1,self.autoencoder_hidden_dim]), dtype=tf.float32) 

			self.w_dec = tf.get_variable('weight_decoder',initializer= tf.truncated_normal([self.autoencoder_hidden_dim,(len(self.feature_index)-1)*self.FM_weight_dim]), dtype=tf.float32) 
			self.b_dec = tf.get_variable('bias_decoder',initializer= tf.truncated_normal([1,(len(self.feature_index)-1)*self.FM_weight_dim]), dtype=tf.float32) 
			
		with tf.variable_scope("LSTM"): 
			self.lstm_cell = [tf.nn.rnn_cell.DropoutWrapper(
						tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_size),
						output_keep_prob = self.lstm_dropout_keep_prob)
						for _ in range(self.lstm_layer_num)]
			self.cell = tf.nn.rnn_cell.MultiRNNCell(self.lstm_cell)
 	
		with tf.variable_scope("block_matching"):	
			self.block_w_enc = tf.get_variable('weight_encoder',initializer= tf.truncated_normal([self.lstm_hidden_size,self.autoencoder_hidden_dim]), dtype=tf.float32) 
			self.block_b_enc = tf.get_variable('bias_encoder',initializer= tf.truncated_normal([1,self.autoencoder_hidden_dim]), dtype=tf.float32) 
			
			self.block_w_dec = tf.get_variable('weight_decoder',initializer= tf.truncated_normal([self.autoencoder_hidden_dim,self.lstm_hidden_size]), dtype=tf.float32) 
			self.block_b_dec = tf.get_variable('bias_decoder',initializer= tf.truncated_normal([1,self.lstm_hidden_size]), dtype=tf.float32) 
			
		

		#item attention to get feature level vectors. 3-D tensor. [batch_size*block_size,feature_num,FM_weight_dim]
		self.feature_vectors = self.Item_attention(self.item_emb,self.item_att_v,self.item_att_w,self.item_att_b)

		#feature attention to get instance vector. 2-D tensor. [batch_size*block_size,(len(self.feature_index)-1)*self.FM_weight_dim]
		attention_vector_self = self.Feature_attention_self(self.feature_vectors,self.feature_att_v,self.feature_att_w,self.feature_att_b)
		attention_vector_previous = self.Feature_attention_previous_block(self.feature_vectors,self.previous_block_att_v,self.previous_block_att_w,self.previous_block_att_b)
		self.instance_vector_real = attention_vector_self+attention_vector_previous #[batch_size*block_size,feature_num*FM_weight_dim]
		
		#add random noise 
		with tf.variable_scope("noise"):
			self.random_noise = tf.get_variable('noise',initializer= tf.truncated_normal([self.batch_size*self.block_size,(len(self.feature_index)-1)*self.FM_weight_dim]), dtype=tf.float32,trainable=False)
		self.instance_vector_real = self.noise*self.random_noise + self.instance_vector_real

		#auto-encoder as GAN generator 
		self.autoencoder_layer = self.Encoder(self.instance_vector_real,self.w_enc,self.b_enc) 
		self.instance_vector_fake = self.Decoder(self.autoencoder_layer,self.w_dec,self.b_dec)
		
		#LSTM to generate real/fake block vector
		
		#generate real block vector
		self.block_vector_real = self.LSTM(self.instance_vector_real,self.cell,"LSTM_real") #2-D tensor [batch_size,lstm_hidden_size]
		self.block_vector_real = tf.layers.batch_normalization(self.block_vector_real, training=self.batch_norm)
		self.previous_block_att_v = tf.reduce_mean(self.block_vector_real,axis=0) #[lstm_hidden_size]. use current real block as attention vector to select features  to instance
		
		#generate fake block vector 
		self.block_vector_fake = self.LSTM(self.instance_vector_fake,self.cell,"LSTM_fake")#2-D tensor [batch_size,lstm_hidden_size]
		self.block_vector_fake = tf.stop_gradient(self.block_vector_fake)
		self.block_matching_hidden = tf.nn.leaky_relu(tf.matmul(self.block_vector_fake,self.block_w_enc)+self.block_b_enc)
		self.block_vector_fake = tf.matmul(self.block_matching_hidden,self.block_w_dec)+self.block_b_dec
		self.block_vector_fake = tf.layers.batch_normalization(self.block_vector_fake, training=self.batch_norm)
		self.block_vector_fake =tf.nn.leaky_relu(self.block_vector_fake)
		
		#GAN to transform instance / block vector
		self.logit_I_real,self.logit_I_fake,self.logit_B_real,self.logit_B_fake = self.GAN_discriminator(self.instance_vector_real,
																										self.instance_vector_fake,
																										self.block_vector_real,
																										self.block_vector_fake) 

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):#for batch normalization 
			
			###generator loss
			# self.generator_loss_instance = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_I_fake, labels=tf.ones_like(self.logit_I_real)),axis = 1)
			# self.generator_loss_block = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_B_fake, labels=tf.ones_like(self.logit_B_real)),axis = 1)
			self.generator_loss_instance = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.instance_vector_fake, labels=tf.nn.sigmoid(self.instance_vector_real)),axis = 1)
			self.generator_loss_block = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.block_vector_fake, labels=tf.nn.sigmoid(self.block_vector_real)),axis = 1)
			self.generator_loss = tf.reduce_mean(self.generator_loss_instance) + tf.reduce_mean(self.generator_loss_block) 
			
			# self.G_train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.generator_loss)
			self.G_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,name='G_optimizer')
			tvars = tf.trainable_variables()
			G_grads, _ = tf.clip_by_global_norm(tf.gradients(self.generator_loss, tvars), 5.0) 
			self.G_train = self.G_optimizer.apply_gradients(zip(G_grads, tvars))
			
			### discriminator loss 
			discriminator_I_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.instance_vector_real, labels=tf.ones_like(self.instance_vector_real)),axis = 1)
			discriminator_I_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.instance_vector_fake, labels=tf.zeros_like(self.instance_vector_fake)),axis = 1)  
			discriminator_B_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.block_vector_real, labels=tf.ones_like(self.block_vector_real)),axis = 1)
			discriminator_B_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.block_vector_fake, labels=tf.zeros_like(self.block_vector_fake)),axis = 1)  
			# discriminator_I_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_I_real, labels=tf.ones_like(self.logit_I_real)),axis = 1)
			# discriminator_I_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_I_fake, labels=tf.zeros_like(self.logit_I_fake)),axis = 1)  
			# discriminator_B_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_B_real, labels=tf.ones_like(self.logit_B_real)),axis = 1)
			# discriminator_B_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_B_fake, labels=tf.zeros_like(self.logit_B_fake)),axis = 1)  
			self.discriminator_loss= tf.reduce_mean(discriminator_I_real_loss)+tf.reduce_mean(discriminator_I_fake_loss)+tf.reduce_mean(discriminator_B_real_loss)+tf.reduce_mean(discriminator_B_fake_loss)
			
			# self.D_train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.discriminator_loss)
			self.D_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,name='D_optimizer')
			D_grads, _ = tf.clip_by_global_norm(tf.gradients(self.discriminator_loss, tvars), 5.0) 
			self.D_train = self.D_optimizer.apply_gradients(zip(D_grads, tvars))

				
		self.instance_total_loss = self.alpha*self.generator_loss_instance #+ self.beta*(discriminator_I_real_loss + discriminator_I_fake_loss)
		self.instance_total_loss_per_block = tf.reduce_sum(tf.reshape(self.instance_total_loss,[-1,self.block_size]),axis = 1)
		self.block_total_loss = self.alpha*self.generator_loss_block + 1.0*self.instance_total_loss_per_block  #+ self.beta*(discriminator_B_real_loss + discriminator_B_fake_loss) 
		self.test1 = self.item_emb
		self.test2 = self.feature_attention_weight
		self.test3 = self.instance_vector_real
		self.test4 = self.instance_total_loss
		
		self.test1 = self.shape1
		self.test2 = self.shape2
		
	#calculate feature level representation
	def Item_attention(self,item_embedding,att_v,att_w,att_b): 

		input_data = tf.reshape(self.data,[-1]) #[batch_size*block_size*intance_dim]
		emb = tf.nn.embedding_lookup(item_embedding,input_data)#[batch_size*block_size*intance_dim,FM_weight_dim] 
		u= tf.tanh(tf.matmul(emb,att_w)+att_b)#[batch_size*block_size*intance_dim,attention_dim]
		vu = tf.matmul(u, att_v) #[batch_size*block_size*intance_dim,1]

		mask = tf.cast(tf.clip_by_value(input_data, 0, 1),tf.float32)#[batch_size*block_size*intance_dim]
		mask = tf.reshape(mask,[-1,1])#[batch_size*block_size*intance_dim,1]

		weight = tf.exp(vu)*mask#[batch_size*block_size*intance_dim,1]

		# item level attention
		input_data = tf.reshape(self.data,[-1,self.instance_dim]) #[batch_size*block_size,intance_dim]
		weight = tf.reshape(weight,[-1,self.instance_dim])#[batch_size*block_size,intance_dim]
		

		feature_vectors = []
		for i in range(len(self.feature_index)-1):
			data_slice = tf.slice(input_data,[0,self.feature_index[i]],[tf.shape(input_data)[0],self.feature_index[i+1]-self.feature_index[i]]) #[batch_size*block_size,related feature length and part of intance_dim]
			weight_slice = tf.slice(weight,[0,self.feature_index[i]],[tf.shape(weight)[0],self.feature_index[i+1]-self.feature_index[i]]) #[batch_size*block_size,related feature length and part of intance_dim]
			weight_sum = tf.reduce_sum(weight_slice,axis = 1)#[batch_size*block_size]
			weight_slice = weight_slice / (tf.reshape(weight_sum,[-1,1]) +1e-5) #[batch_size*block_size,related feature length and part of intance_dim]  # to aviod nan, add smoothing
			self.item_attention_weight = weight_slice
			weight_slice = tf.expand_dims(weight_slice,-1)
			emb = tf.nn.embedding_lookup(item_embedding,data_slice) #[batch_size*block_size,related feature length and part of intance_dim,FM_weight_dim]
			emb = weight_slice * emb #[batch_size*block_size,related feature length and part of intance_dim,FM_weight_dim]
			emb = tf.reduce_sum(emb,1)
			feature_vectors.append(emb)

		feature_vectors = tf.stack(feature_vectors,axis = 1)	
		feature_vectors = tf.layers.batch_normalization(feature_vectors, training=self.batch_norm)
		feature_vectors = tf.nn.leaky_relu(feature_vectors)
		return feature_vectors 

	def Feature_attention_self(self,data,att_v,att_w,att_b):
		
		input_data = tf.reshape(data,[-1,tf.shape(data)[-1]]) #[batch_size*block_size*feature_num,FM_weight_dim]
		u= tf.tanh(tf.matmul(input_data,att_w)+att_b)#[batch_size*block_size*feature_num,attention_dim]
		vu = tf.matmul(u, att_v) #[batch_size*block_size*feature_num,1]
		weight = tf.exp(vu)#[batch_size*block_size*feature_num,1]

		# feature level attention
		weight = tf.reshape(weight,[-1,tf.shape(data)[1]]) #[batch_size*block_size,feature_num]
		weight_sum = tf.reduce_sum(weight,axis = 1)#[batch_size*block_size]
		weight = weight / (tf.reshape(weight_sum,[-1,1]) +1e-5) #[batch_size*block_size,feature_num]  # to aviod nan, add smoothing
		self.feature_attention_weight = weight
		weight = tf.expand_dims(weight,-1)#[batch_size*block_size,feature_num,1]

		instance_vector = weight * data #[batch_size*block_size,feature_num,FM_weight_dim]
		self.shape1 = tf.shape(instance_vector)
		# instance_vector = tf.reduce_sum(instance_vector,1) #[batch_size*block_size,FM_weight_dim] # sum all weights
		instance_vector = tf.reshape(instance_vector,[-1,(len(self.feature_index)-1)*self.FM_weight_dim]) # concatenate all feature vectors [batch_size*block_size,feature_num*FM_weight_dim] 
		instance_vector = tf.layers.batch_normalization(instance_vector, training=self.batch_norm)
		instance_vector = tf.nn.leaky_relu(instance_vector)
		

		return instance_vector

	def Feature_attention_previous_block(self,data,att_v,att_w,att_b):
		att_v = tf.reshape(att_v,[-1,1])
		input_data = tf.reshape(data,[-1,tf.shape(data)[-1]]) #[batch_size*block_size*feature_num,FM_weight_dim]
		u= tf.tanh(tf.matmul(input_data,att_w)+att_b)#[batch_size*block_size*feature_num,attention_dim]
		vu = tf.matmul(u, att_v) #[batch_size*block_size*feature_num,1]
		weight = tf.exp(vu)#[batch_size*block_size*feature_num,1]

		# feature level attention
		weight = tf.reshape(weight,[-1,tf.shape(data)[1]]) #[batch_size*block_size,feature_num]
		weight_sum = tf.reduce_sum(weight,axis = 1)#[batch_size*block_size]
		weight = weight / (tf.reshape(weight_sum,[-1,1]) +1e-5) #[batch_size*block_size,feature_num]  # to aviod nan, add smoothing
		weight = tf.expand_dims(weight,-1)#[batch_size*block_size,feature_num,1]

		instance_vector = weight * data #[batch_size*block_size,feature_num,FM_weight_dim]
		self.shape2 = tf.shape(instance_vector)
		# instance_vector = tf.reduce_sum(instance_vector,1) #[batch_size*block_size,FM_weight_dim] # sum all weights
		instance_vector = tf.reshape(instance_vector,[-1,(len(self.feature_index)-1)*self.FM_weight_dim]) # concatenate all feature vectors [batch_size*block_size,feature_num*FM_weight_dim] 
		instance_vector = tf.layers.batch_normalization(instance_vector, training=self.batch_norm)
		instance_vector = tf.nn.leaky_relu(instance_vector)
		
		return instance_vector

	def Encoder(self,data,w_enc,b_enc): 
		hidden = tf.nn.leaky_relu(tf.matmul(data,w_enc)+b_enc)#[batch_size*block_size,autoencoder_hidden_dim]
		return hidden 

	def Decoder(self,data,w_dec,b_dec): 
		instance_vector_fake = tf.nn.leaky_relu(tf.matmul(data,w_dec)+b_dec) 
		return instance_vector_fake #[batch_size*block_size, (len(self.feature_index)-1)*self.FM_weight_dim]
	
	def LSTM(self,data,cell,variable_scope):
		data = tf.reshape(data,[-1,self.block_size,(len(self.feature_index)-1)*self.FM_weight_dim]) #[batch_size,block_size,(len(self.feature_index)-1)*self.FM_weight_dim]
		state = cell.zero_state(self.batch_size,tf.float32)
		with tf.variable_scope(variable_scope): 
			for time_step in range(self.block_size):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				cell_output,state = cell(data[:,time_step,:],state) # the last step output is regarded as the block vector
		
		# cell_output = tf.layers.batch_normalization(cell_output, training=self.batch_norm)
		return cell_output

	def GAN_discriminator(self,instance_vector_real,instance_vector_fake,block_vector_real,block_vector_fake):
		with tf.variable_scope("instance"): 
			self.w_instance_hidden = tf.get_variable('w1',initializer= tf.truncated_normal([(len(self.feature_index)-1)*self.FM_weight_dim,self.gan_hidden_dim]), dtype=tf.float32)  
			self.b_instance_hidden = tf.get_variable('b1',initializer= tf.truncated_normal([1,self.gan_hidden_dim]), dtype=tf.float32) 

			self.w_instance_output = tf.get_variable('w2',initializer= tf.truncated_normal([self.gan_hidden_dim,1]), dtype=tf.float32) 
			self.b_instance_output = tf.get_variable('b2',initializer= tf.truncated_normal([1,1]), dtype=tf.float32) 

		with tf.variable_scope("block"): 
			self.w_block_hidden = tf.get_variable('w1',initializer= tf.truncated_normal([self.lstm_hidden_size,self.gan_hidden_dim]), dtype=tf.float32) 
			self.b_block_hidden = tf.get_variable('b1',initializer= tf.truncated_normal([1,self.gan_hidden_dim]), dtype=tf.float32) 

			self.w_block_output = tf.get_variable('w2',initializer= tf.truncated_normal([self.gan_hidden_dim,1]), dtype=tf.float32) 
			self.b_block_output = tf.get_variable('b2',initializer= tf.truncated_normal([1,1]), dtype=tf.float32) 

		logit_I_real = self.discriminator_instance(tf.nn.sigmoid(instance_vector_real))
		logit_I_fake = self.discriminator_instance(tf.nn.sigmoid(instance_vector_fake))
		logit_B_real = self.discriminator_block(tf.nn.sigmoid(block_vector_real))
		logit_B_fake = self.discriminator_block(tf.nn.sigmoid(block_vector_fake))
		return logit_I_real,logit_I_fake,logit_B_real,logit_B_fake

		

	def discriminator_instance(self,data):
		instance = tf.reshape(data,[-1,(len(self.feature_index)-1)*self.FM_weight_dim])
		hidden_layer = tf.nn.leaky_relu(tf.matmul(instance, self.w_instance_hidden) + self.b_instance_hidden)
		output = tf.matmul(hidden_layer,self.w_instance_output)+self.b_instance_output
		return output

	def discriminator_block(self,data):
		hidden_layer = tf.nn.leaky_relu(tf.matmul(data, self.w_block_hidden) + self.b_block_hidden)
		output = tf.matmul(hidden_layer,self.w_block_output)+self.b_block_output
		return output






