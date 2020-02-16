import numpy as np
import tensorflow as tf
import utils

class CNN(object):
    def __init__(self, n_classes, img_height, img_width, img_channel, device_name='/cpu:0'):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, img_height, img_width, img_channel], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None,25088], name='input_y')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

        with tf.device(device_name):
            self.input_image = tf.reshape(self.input_x, [-1,img_height,img_width,img_channel])
            self.h_pool_1=self.layer(1,self.input_x,f_s=[5,5,img_channel,8])
            self.h_pool_2=self.layer(2,self.h_pool_1,f_s=[3,3,8,16])
            self.h_pool_3=self.layer(3,self.h_pool_2,f_s=[3,3,16,32])

            num_total_unit = self.h_pool_3.get_shape()[1:4].num_elements()
            print(' num_total_unit  %d' % num_total_unit)
            self.h_pool_3_flat = tf.reshape(self.h_pool_3, shape=[-1, num_total_unit])
            print(self.h_pool_3_flat.shape)

            with tf.name_scope('fc_layer_1'):
                self.h_fc_1 = self.fc_layer(self.h_pool_3_flat, num_total_unit, 128, activation_function=tf.nn.relu)

            with tf.name_scope('dropout'):
                 self.h_drop = tf.nn.dropout(self.h_fc_1, keep_prob=self.dropout_keep_prob, name='h_drop')

            with tf.name_scope('fc_layer_2'):
                self.output = self.fc_layer(self.h_drop, 128, n_classes, activation_function=None)

        with tf.device('/cpu:0'):
          

            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.square( self.input_y - self.h_pool_3_flat))
 

    def w_variable(self, shape):
        return tf.Variable(initial_value=tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1), dtype=tf.float32, name='W')

    def b_variable(self, shape):
        return tf.Variable(initial_value=tf.constant(value=0.1, shape=shape), dtype=tf.float32, name='b')

    def conv2d(self, x, W, stride, padding='SAME' ,name='conv'):
        return tf.nn.conv2d(input=x, filter=W, strides=[1,stride,stride,1], padding=padding, name=name)

    def max_pool(self, x, ksize, stride, padding='VALID',name='max_pool'):
        return tf.nn.max_pool(value=x, ksize=[1,ksize,ksize,1], strides=[1,stride,stride,1], padding=padding, name=name)
 
    def conv(self,x,f_s,stride=1,padding='SAME' ,name='conv'):
        return self.conv2d(x, self.w_variable(shape=f_s),stride,padding,name)

    def layer(self,step, input ,f_s,ksize=2,c_stride=1,m_stride=2,c_padding='SAME',m_padding='VALID' ):
        cstep=str(step)
        c_scope_name='conv_layer_'+cstep
        m_scope_name='pooling_layer_'+cstep
        with tf.name_scope(c_scope_name):
            h_conv = self.conv(x=input, f_s=f_s, stride=c_stride, padding=c_padding,name='conv_'+cstep)
            h_conv = tf.nn.relu(features=h_conv, name='relu_conv_'+cstep)
        with tf.name_scope(m_scope_name):
            h_pool = self.max_pool(x=h_conv, ksize=2, stride=2, padding='SAME',name='max_pool_'+cstep )   
        return h_pool
    




    def fc_layer(self, x, in_size, out_size, activation_function=None):
        w = self.w_variable(shape=[in_size, out_size])
        b = self.b_variable(shape=[out_size])
        z = tf.nn.xw_plus_b(x, w, b, name='Wx_plus_b')
        if activation_function is None:
            outputs = z
        else:
            outputs = activation_function(z)
        return outputs

    def op(self,rate):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate=rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return train_op,global_step
