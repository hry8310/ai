# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq 
import numpy as np
import time
import os

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

class RNN(object):
    def __init__(self, vocab_size, num_seqs=64, num_steps=50,
                 rnn_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, training=True, train_keep_prob=0.5, use_embedding=False, embedding_size=128,mode='lstm'):
        if training is True:
            num_seqs, num_steps = num_seqs, num_steps
        else:
            num_seqs, num_steps = 1, 1
    
        self.mode=mode
        self.vocab_size =vocab_size 
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self._inputs()
        self._model(training)
        self.saver = tf.train.Saver()

    def _inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            if self.use_embedding is True:
                print('one_hotinnnnnnnnnnnnnnn')
                self.rnn_inputs = tf.one_hot(self.inputs, self.vocab_size)
            else:
                with tf.device("/cpu:0"):
                    self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
                    self.rnn_inputs = tf.nn.embedding_lookup(self.embedding, self.inputs)

    def _model(self ,training ):
        # 创建单个cell并堆叠多层
        def get_a_cell(rnn_size, keep_prob):
            if self.mode == 'rnn':
                fn = rnn.RNNCell
            elif self.mode == 'gru':
                fn = rnn.GRUCell
            elif self.mode == 'lstm':
                fn = rnn.LSTMCell
            else :
                fn = rnn.NASCell
            _cell = fn(rnn_size)
            drop = rnn.DropoutWrapper(_cell, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('rnn_model'):
            cell = rnn.MultiRNNCell(
                [get_a_cell(self.rnn_size, self.keep_prob) for _ in range(self.num_layers)] , state_is_tuple=True
            )
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)
             
            softmax_w = tf.get_variable("softmax_w",
                                        [self.rnn_size, self.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
            inputs = tf.split(self.rnn_inputs, self.num_steps, 1)
            #self.rnn_inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

            def loop(prev, _):
                prev = tf.matmul(prev, softmax_w) + softmax_b
                prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                return rnn.embedding_lookup(self.embedding, prev_symbol)

            self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.rnn_inputs, initial_state=self.initial_state)
            #self.rnn_outputs, self.final_state = legacy_seq2seq.rnn_decoder(self.rnn_inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm') 

            print(self.rnn_outputs.shape)
            seq_output = tf.concat(self.rnn_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.rnn_size])
            print(seq_output.shape)
            print(x.shape)


            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

        with tf.name_scope('loss'):
            y_reshaped = tf.reshape(self.targets, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))
  
    def summary(self, sum_dir):
        for var in tf.trainable_variables():
            tf.summary.histogram(name=var.name, values=var)
        tf.summary.scalar('loss', tf.reduce_mean(self.loss))
        tf.summary.scalar('logits', tf.reduce_mean(self.logits))
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(sum_dir, tf.get_default_graph())
        return summary_op,summary_writer

    def train(self,  batch_generator, sum_dir,max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            summary_op,summary_writer=self.summary(sum_dir)
            sess.run(tf.global_variables_initializer())
            # Train network
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()

                if self.mode == 'lstm' :
                    feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob
                        }
                    for i, (c, h) in enumerate(self.initial_state):
                        feed[c] = new_state[i].c
                        feed[h] = new_state[i].h
                else:
                    feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
               
                batch_loss, summaries,ynew_state, _ = sess.run([self.loss,summary_op,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                    summary_writer.add_summary(summaries,step)
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))  # for prime=[]
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, vocab_size)
        samples.append(c)

        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        print(checkpoint)
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
