#! /usr/local/bin/python

# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

def model(x,w1,w2,b1,b2):
    a = tf.matmul(x,w1)
    b = tf.add(a,b1)
    c = tf.sigmoid(b)
    hidden = tf.sigmoid(tf.add(tf.matmul(x,w1),b1))

    out = tf.nn.softmax(tf.add(tf.matmul(hidden,w2),b2))
    return out

x = tf.placeholder("float",[4,4])
w1 = tf.Variable(tf.random_normal([4,2]),name='w1')
w2 = tf.Variable(tf.random_normal([2,4]),name = 'w2')
b1 = tf.Variable(tf.random_normal([2]),name = 'b1')
b2 = tf.Variable(tf.random_normal([4]),name = 'b2')

pred = model(x,w1,w2,b1,b2)
cost = tf.reduce_sum(tf.pow(pred-x,2))

optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    input_data = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]],float)
    for i in xrange(10000):
        sess.run(optimizer,feed_dict={x:input_data})

    res = sess.run(pred,feed_dict={x:input_data})
    print res
    
    index = np.argmax(res,1)
    print index
    for i in xrange(4):
        tmp = np.zeros((4,))
        tmp[index[i]]=1.
        print res[i]
        print tmp