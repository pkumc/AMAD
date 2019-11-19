import tensorflow as tf
import numpy as np

class AnomalyNetConfig(object):
	batch_size = 64
	X_emb_dim = 128
	domain_emb_dim = 256
	hidden_dim = 64
	learning_rate = 0.01
	epoch = 10
class AnomalyNet(object):
	def __init__(self,
				batch_size,
				X_emb_dim,
				domain_emb_dim,
				hidden_dim,
				learning_rate): 
		self.batch_size = batch_size
		self.X_emb_dim = X_emb_dim
		self.domain_emb_dim = domain_emb_dim
		self.hidden_dim = hidden_dim
		self.learning_rate = learning_rate
		self.X = tf.placeholder(dtype=tf.float32,shape=[None, X_emb_dim], name='X')
		self.y = tf.placeholder(dtype=tf.float32,shape=[None, 1], name='y')
		self.label = tf.placeholder(dtype=tf.float32,shape=[None, 1], name='label')

		with tf.variable_scope('gate'): #initialize domain vector
			self.domain = tf.get_variable("domain",[domain_emb_dim],initializer=tf.random_normal_initializer(mean=0, stddev=1))
		self.X_filtered, self.y_filtered, self.loss_ave = activation_gate()
		self.is_empty = tf.equal(tf.size(self.X_filtered), 0)
		if not self.is_empty:
			self.domain = update_gate()
			self.pred = task_net()
			self.loss = loss_ave + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( logits=self.pred, labels=self.y),name='loss')
			optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
			self.train_op = optimizer.minimize(self.loss,name='train_op')  
		
	def activation_gate(self):
		domain_expend = tf.ones([self.batch_size, 1]) * self.domain
		concant = tf.concat([self.X, domain_expend], 1) #concatenate input and domain vector for each batch
		with tf.variable_scope('activation'): #initialize domain vector
			activation_w = tf.get_variable("w",[self.X_emb_dim+self.domain_emb_dim,1],initializer=tf.random_normal_initializer(mean=0, stddev=1))
			activation_b = tf.get_variable("b",[1],initializer=tf.random_normal_initializer(mean=0, stddev=1))
			mask = tf.nn.sigmoid(tf.add(tf.matmul(concant,activation_w),activation_b))
			mask = tf.greater(mask, 0.5)# boolean, check whether the activation is greater than 0.5
			X_filtered = tf.boolean_mask(self.X, mask)#filtered X
			y_filtered = tf.boolean_mask(self.y, mask)#filtered y
			loss_ave = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( logits=mask, labels=self.label),name='loss')
		return X_filtered, y_filtered, loss_ave

	def update_gate(self):
		X_ave = tf.reduce_mean(self.X_filtered, 0) # average X_filtered in the first dimension and make it to a 1d tensor
		concant = tf.concat([X_ave, self.domain], 1) #concatenate input and domain vector for each batch
		with tf.variable_scope('update'): #initialize domain vector
			update_w = tf.get_variable("w",[self.input_emb_dim+self.domain_emb_dim,self.domain_emb_dim],initializer=tf.random_normal_initializer(mean=0, stddev=1))
			update_b = tf.get_variable("b",[self.domain_emb_dim],initializer=tf.random_normal_initializer(mean=0, stddev=1))
			self.domain = tf.nn.l2_normalize(tf.add(tf.matmul(concant,update_w),update_b))


	def task_net(self):
		with tf.variable_scope('mlp'): 
				w = tf.get_variable("w",[self.X_emb_dim, self.hidden_dim],initializer=tf.random_normal_initializer(mean=0, stddev=1))
				b = tf.get_variable("b",[self.hidden_dim],initializer=tf.random_normal_initializer(mean=0, stddev=1))
				MLP1 = tf.add(tf.matmul(X_filtered, w), b)  
				MLP1 = tf.nn.relu(MLP1)

		with tf.variable_scope('out'): 
				w = tf.get_variable("w",[self.hidden_dim, 1],initializer=tf.random_normal_initializer(mean=0, stddev=1))
				b = tf.get_variable("b",[1],initializer=tf.random_normal_initializer(mean=0, stddev=1))
				out = tf.add(tf.matmul(X_filtered, w), b)  
				# out = tf.nn.sigmoid(out)
		return out



# if __name__ == '__main__':










	