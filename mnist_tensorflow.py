# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 23:10:06 2018

@author: Leo
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
 
batch_size = 100
n_batch = mnist.train.num_examples//batch_size
x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)
 #第一层
w1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,w1)+b1)
#dropout方法，keep_prob表示有多少比例的神经元在工作
L1_drop = tf.nn.dropout(L1,keep_prob)


#第二层
w2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,w2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)


w3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,w3)+b3)
L3_drop = tf.nn.dropout(L3,keep_prob)


w4 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop,w4)+b4)


#loss = tf.reduce_mean(tf.square(y-L4))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for i in range(19):
        for o in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={y:batch_ys,x:batch_xs,keep_prob:0.7})
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:0.7})
        train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:0.7})
        if i%3==0:
            print('第'+str(i)+'次训练准确率为：'+str(train_acc)+'测试准确率为：'+str(test_acc))
    