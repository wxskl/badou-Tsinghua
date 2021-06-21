#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2,input3)
mul = tf.multiply(input1,intermed)

with tf.Session() as sess:
    result = sess.run([mul,intermed]) #需要获取的多个tensor值，在op的一次运行中一起获得（而不是逐个去获取tensor）。
    print(result)

