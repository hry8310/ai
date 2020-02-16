# -*- coding: utf-8 -*-

import sys
import os
import time
import datetime
import numpy as np
import tensorflow as tf
import shutil
import utils 
from model import CNN
from Config import RunConfig








class Train(object):
            
    def __init__(self):
        self.config=RunConfig()
        self.loaddata()
        self.ready()
        
    def loaddata(self):
        print('Loading data..')
        x_path, y,y2 = utils.get_img(self.config.train_data_dir)
    
        # split train/dev set
        split_index = -int(float(len(y)) * self.config.dev_sample_percentage)
        self.x_path_train, self.x_path_dev = x_path[:split_index], x_path[split_index:]
        self.y_train, self.y_dev = y[:split_index], y[split_index:]
        self.y_train2, self.y_dev2 = y2[:split_index], y2[split_index:]
        del x_path, y,y2
    
        self.x_dev = []
        for i in range(len(self.x_path_dev)):
            img_data = utils.img_resize(img_path=self.x_path_dev[i], img_height=self.config.img_height, img_width=self.config.img_width)
            img_data = utils.rgb2gray(img_data)
            self.x_dev.append(img_data)
        self.x_dev = np.array(self.x_dev)
        self.y_dev = np.array(self.y_dev) 
        
    def summary( self,t ):
        tf.summary.scalar('loss', tf.reduce_mean(self.cnn.loss))
        summary_op = tf.summary.merge_all()
        summary_dir = os.path.join(self.out_dir, 'summaries', t)
        summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())
        return summary_op,summary_writer
    
    def ready(self):
    	
        cdir=os.path.join(os.curdir, 'log','ck')
        if os.path.exists(cdir):
            shutil.rmtree(cdir)
        self.out_dir = os.path.abspath(cdir)
        
        checkpoint_dir = os.path.join(self.out_dir, 'checkpoints')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def iter(self,x_batch, y_batch,summary,prob,writer=None,train_op=None):
        if train_op is None : 
            train_op=self.global_step
        print('iter.........') 
        yy_batch=utils.get_m_arr(y_batch)
        print(yy_batch.shape)
        feed_dict = {
            self.cnn.input_x: x_batch,
            self.cnn.input_y: yy_batch,
            self.cnn.dropout_keep_prob: prob
        }
        _, step, summaries, loss  = self.sess.run(
            [train_op, self.global_step, summary, self.cnn.loss],
            feed_dict)
        timestr = datetime.datetime.now().isoformat()
        print('{}: step {} '.format(timestr, step))
        if writer:
            writer.add_summary(summaries, step) 
        return loss    
        
    def run(self):    
        print('run begin...1111\n')
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.config.allow_soft_placement,
                log_device_placement=self.config.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                self.cnn = CNN(
                    n_classes=self.y_train.shape[1],
                    img_height=self.config.img_height,
                    img_width=self.config.img_width,
                    img_channel=self.config.img_channels,
                    device_name=self.config.device_name
                )
                self.train_op ,self.global_step = self.cnn.op(self.config.learning_rate)
        
                
        
                tf.summary.image('input_image', self.cnn.input_image, max_outputs=self.config.batch_size)
        
                for var in tf.trainable_variables():
                    tf.summary.histogram(name=var.name, values=var)
        
                train_summary_op,train_summary_writer=self.summary('train')
                dev_summary_op,dev_summary_writer=self.summary('dev')
                
        
                # checkpointing, tensorflow assumes this directory already existed, so we need to create it
                
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.config.num_checkpoints)
        
                
        
                print('run begin...i222222\n')
        
                ### training loop
                # train loop, for each batch
                self.sess.run(tf.global_variables_initializer())
                batches = utils.batch_iter(batch_size=self.config.batch_size, 
                          num_epochs=self.config.num_epochs, img_path_list=self.x_path_train, label_list=self.y_train,label_list2=self.y_train2,
                          img_height=self.config.img_height, img_width=self.config.img_width)
                best_acc_val=0.0
                for x_batch, y_batch,y_batch2 in batches:
                    print(y_batch2)
                    t_loss =self.iter(x_batch, y_batch2,train_summary_op,self.config.dropout_keep_prob, writer=train_summary_writer,train_op=self.train_op)
                    current_step = tf.train.global_step(self.sess, self.global_step)
                    
                    if current_step % self.config.checkpoint_every == 0:
                        print(self.checkpoint_prefix)
                        path = saver.save(sess=self.sess, save_path=self.checkpoint_prefix, global_step=self.global_step)
                    if current_step == 10000:
                        break

# end
train =Train()
train.run()
print('over..................')
