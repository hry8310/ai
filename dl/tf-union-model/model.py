import numpy as np
import shmode as shm
import lenmode as lenm
import tensorflow as tf



class UCNN(object):
    def __init__(self, n_classes, img_height, img_width, img_channel, device_name='/cpu:0'):
        self.sh=shm.CNN( n_classes, img_height, img_width, img_channel, device_name)
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='input_y')
        self.len=lenm.CNN( n_classes, img_height, img_width, img_channel, device_name)
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
        self.un_out=tf.concat([self.len.h_drop_2 , self.sh.h_drop],axis=-1)


        with tf.device(device_name):

            with tf.name_scope('un_fc_layer_2'):
                self.output = self.fc_layer(self.un_out, 192, n_classes, activation_function=None)

        with tf.device('/cpu:0'):
            with tf.name_scope('prediction'):
                self.y_pred = tf.argmax(input=self.output, axis=1, name='y_pred')

            with tf.name_scope('loss'):
                self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.output)

            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.y_pred, tf.argmax(self.input_y, axis=1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

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
