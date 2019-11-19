#!/usr/bin/python
# -*- coding:  utf-8 -*-
'''
Tensorflow implementation to learn instance vector representation

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


class Autoencoder(object):

    def __init__(self,hidden_dim,block_size,instance_dim,random_seed=2018):
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.instance_dim = instance_dim
        self.random_seed = random_seed

        self.data = tf.placeholder(tf.float32, shape=[None,self.block_size,self.instance_dim], name="data") 
        
        with tf.variable_scope("autoencoder"):
            self.w_enc = tf.get_variable('weight_encoder',initializer= tf.ones([self.instance_dim,self.hidden_dim]), dtype=tf.float32) 
            self.b_enc = tf.get_variable('bias_encoder',initializer= tf.ones([1,self.hidden_dim]), dtype=tf.float32) 

            self.w_dec = tf.get_variable('weight_decoder',initializer= tf.ones([self.hidden_dim,self.instance_dim]), dtype=tf.float32) 
            self.b_dec = tf.get_variable('bias_decoder',initializer= tf.ones([1,self.instance_dim]), dtype=tf.float32) 
        self.encoder()
        self.decoder()
        self._init_session() 
    
    def _init_session(self): 
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True 
        self.sess = tf.Session(config=config)
        # self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)
            
    def encoder(self):
        #input to hidden layer
        # input are 3-D: batch_size, block_size, instance_dim
        # convert to 3-D:block_size, batch_size, instance_dim
        # after transformation: block_size, batch_size, hidden_dim
        data = tf.transpose(self.data,[1,0,2])
        w_enc = tf.reshape(self.w_enc,[-1,tf.shape(self.w_enc)[0],tf.shape(self.w_enc)[1]])
        w_enc = tf.tile(w_enc,[tf.shape(data)[0],1,1])
        b_enc = tf.reshape(self.b_enc,[-1,tf.shape(self.b_enc)[0],tf.shape(self.b_enc)[1]])
        b_enc = tf.tile(b_enc,[tf.shape(data)[0],tf.shape(data)[1],1])
        self.hidden  = tf.matmul(data,w_enc)+b_enc
        self.hidden = tf.nn.sigmoid(self.hidden)
        # self.hidden = b_enc

    def decoder(self):
        #hidden layer to output 
        #hidden layer 3-D:block_size, batch_size, hidden_dim
        #output 3-D: batch_size, block_size, instance_dim
        w_dec = tf.reshape(self.w_dec,[-1,tf.shape(self.w_dec)[0],tf.shape(self.w_dec)[1]])
        w_dec = tf.tile(w_dec,[tf.shape(self.hidden)[0],1,1])
        b_dec = tf.reshape(self.b_dec,[-1,tf.shape(self.b_dec)[0],tf.shape(self.b_dec)[1]])
        b_dec = tf.tile(b_dec,[tf.shape(self.hidden)[0],tf.shape(self.hidden)[1],1])

        self.output = tf.matmul(self.hidden,w_dec)+b_dec
        self.output = tf.nn.sigmoid(self.output)
        self.output = tf.transpose(self.output,[1,0,2]) #get back to 3-D:  batch_size, block_size, hidden_dim
        


    def train(self,data): 
        self.graph = tf.Graph()
        with self.graph.as_default():   
            feed_dict = {self.data: data}
            data = self.sess.run((self.data,self.output), feed_dict=feed_dict)
            print("result is:",data[0],data[0].shape,data[1],data[1].shape)#data[2],data[2][0].shape
        
    def close(self):#shut down the session
        self.sess.close()

if __name__ == '__main__': 
    data = [[[2,0,1,1,2],[1,2,1,1,0]],
            [[1,0,0,3,2],[0,0,2,0,0]]]
    data = np.asarray(data)
    model = Autoencoder(10,2,5)
    model.train(data)
    model.close()    
