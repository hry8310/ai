import os
import time
import shutil
import numpy as np
import tensorflow as tf
import lib.utils as utils
from tqdm import tqdm
from lib.dataset import Dataset
from lib.model import Yolo3 
from lib.config import cfg


class Train(object):
    def __init__(self):
        self.anchor_per_scale    = cfg.Anchor_scale
        self.classes             = utils.read_class_names(cfg.Classes)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfg.Lr_begin
        self.learn_rate_end      = cfg.Lr_end
        self.first_stage_epochs  = cfg.First_ep
        self.second_stage_epochs = cfg.Second_ep
        self.warmup_periods      = cfg.Warm_ep
        self.initial_weight      = cfg.Init_weight
        self.save_weight_dir      = cfg.Save_weight_dir
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.Avg_decay
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.steps_per_period    = len(self.trainset)
        self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        with tf.name_scope('cfg_input'):
            self.trainable     = tf.placeholder(dtype=tf.bool, name='training')

        self.model = Yolo3( self.trainable)
        self.net_var = tf.global_variables()
        self.model.model_loss()
        
        with tf.name_scope('model_input'):
            self.input_data  = self.model.input_data 

        with tf.name_scope("model_loss"):
            self.label_sbox  = self.model.label_sbox 
            self.label_mbox  = self.model.label_mbox 
            self.label_lbox  = self.model.label_lbox 
            self.true_sboxes = self.model.true_sboxes 
            self.true_mboxes = self.model.true_mboxes 
            self.true_lboxes = self.model.true_lboxes
            self.loss = self.model.last_loss
            self.xywh_loss = self.model.xywh_loss
            self.conf_loss = self.model.conf_loss
            self.prob_loss = self.model.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate",      self.learn_rate)
            tf.summary.scalar("xywh_loss",  self.xywh_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = "./data/log/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer  = tf.summary.FileWriter(logdir, graph=self.sess.graph)


    def train(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = cfg.Init_ep

        for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step],feed_dict={
                                                self.input_data:   train_data[0],
                                                self.label_sbox:  train_data[1],
                                                self.label_mbox:  train_data[2],
                                                self.label_lbox:  train_data[3],
                                                self.true_sboxes: train_data[4],
                                                self.true_mboxes: train_data[5],
                                                self.true_lboxes: train_data[6],
                                                self.trainable:    True,
                })

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" %train_step_loss)

            for test_data in self.testset:
                test_step_loss = self.sess.run( self.loss, feed_dict={
                                                self.input_data:   test_data[0],
                                                self.label_sbox:  test_data[1],
                                                self.label_mbox:  test_data[2],
                                                self.label_lbox:  test_data[3],
                                                self.true_sboxes: test_data[4],
                                                self.true_mboxes: test_data[5],
                                                self.true_lboxes: test_data[6],
                                                self.trainable:    False,
                })

                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "yolov3_loss=%.4f.ckpt" % test_epoch_loss
            ckpt_file = self.save_weight_dir+ckpt_file 
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)



if __name__ == '__main__': Train().train()




