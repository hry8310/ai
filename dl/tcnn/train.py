#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from model import Config, TextCNN
from utils import Text

text=Text()

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'logs'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


class Train(object):
	  
    def __init__(self):
        
        print()

    def ready(self):
        tensorboard_dir = 'tensorboard/textcnn'
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
    
        tf.summary.scalar("loss", model.loss)
        tf.summary.scalar("accuracy", model.acc)
        self.summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(tensorboard_dir)
        self.saver = tf.train.Saver()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.session = tf.Session()    
        self.session.run(tf.global_variables_initializer())
      
        
    def train(self):
        self.ready()        
        x_train, y_train = text.do_file(train_dir, config.seq_length)
         
    
        self.writer.add_graph(self.session.graph)
    
        total_batch = 1  # 总批次
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
    
        flag = False
        batch_train = text.batch_iter(x_train, y_train, config.batch_size)
        for _ in range(config.print_per_batch):
            for x_batch, y_batch in batch_train:
                feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
                feed_dict[model.keep_prob] = 1.0
                summary ,_, loss_train, acc_train = self.session.run([self.summary,model.optim,model.loss, model.acc], feed_dict=feed_dict)
                self.writer.add_summary(summary, total_batch)

        current_step = tf.train.global_step(self.session, model.global_step)    
        self.saver.save(sess=self.session, save_path=save_path,global_step=model.global_step)
    
    
    
    
config = Config()
text.build_vocab(train_dir, vocab_dir)
config.vocab_size=text.get_vocab_size()
model = TextCNN(config)
tr=Train() 
tr.train()
