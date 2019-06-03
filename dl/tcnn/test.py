# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from model import Config, TextCNN
from utils import Text

try:
    bool(type(unicode))
except NameError:
    unicode = str

text=Text()

base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation-782')  # 最佳验证结果保存路径


config = Config()
categories, cat_to_id = text.read_category()
words, word_to_id =text.read_vocab(vocab_dir)
config.vocab_size = len(words)
print(config.vocab_size)
model = TextCNN(config)
session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

def predict( content):
    data = [word_to_id[x] for x in content if x in word_to_id]

    feed_dict = {
        model.input_x: kr.preprocessing.sequence.pad_sequences([data], config.seq_length),
        model.keep_prob: 1.0
    }

    y_pred_cls = session.run(model.y_pred_cls, feed_dict=feed_dict)
    return categories[y_pred_cls[0]]


if __name__ == '__main__':
    test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
    test_demo2 = ['而到了联赛中，郭艾伦和于德豪之间的对位则充满了火药味。上赛季在辽宁男篮客场与深圳男篮首节的一次进攻中，郭艾伦持球，裁判已经吹罚了深圳队犯规',
                 '在保证服务品质尽可能统一方面，可以看到的是，好未来似乎做了更多的努力。比如加强教研，保证课程内容的统一；启动双师等项目，通过扩大核心教师的产能以适应不断增长的学生数量。相比之下，依靠名师文化起家的新东方，虽然也在不断加大教研和技术的投入，但是显然不够，因为快速扩张带来的教师等优质服务人员不足的问题会更加突出。过度包装名师、降低教师的选拔标准等问题也在许多机构中被曝光出来']
    for i in test_demo2:
        print(predict(i))
