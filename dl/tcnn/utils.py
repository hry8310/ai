# coding: utf-8
import os 
import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
class Text(object):
    def __init__(self,vocab_size=5000):
        self.vocab_size=vocab_size

    def load_file(self,filename):
        """读取文件数据"""
        contents, labels = [], []
        with open(filename) as f:
            for line in f:
                try:
                    label, content = line.strip().split('\t')
                    if content:
                        contents.append(list(content))
                        labels.append(label)
                except:
                    pass
        return contents, labels
    
    
    def _build_vocab(self,train_dir, vocab_dir):
        """根据训练集构建词汇表，存储"""
        data_train, _ = self.load_file(train_dir)
    
        all_data = []
        for content in data_train:
            all_data.extend(content)
    
        counter = Counter(all_data)
        count_pairs = counter.most_common(self.vocab_size - 1)
        words, _ = list(zip(*count_pairs))
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        words = ['<PAD>'] + list(words)
        open(vocab_dir, mode='w').write('\n'.join(words) + '\n')
    
    
    def read_vocab(self,vocab_dir):
        """读取词汇表"""
        # words = open(vocab_dir).read().strip().split('\n')
        with open(vocab_dir) as fp:
            # 如果是py2 则每个值都转化为unicode
            words = [_.strip() for _ in fp.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
        return words, word_to_id
    
    
    def read_category(self):
        """读取分类目录，固定"""
        categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    
        categories = [x for x in categories]
    
        cat_to_id = dict(zip(categories, range(len(categories))))
    
        return categories, cat_to_id
   
    def build_vocab(self,train_dir,vocab_dir):
        if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
            self._build_vocab(train_dir, vocab_dir)
        self.categories, self.cat_to_id = self.read_category()
        self.words, self.word_to_id = self.read_vocab(vocab_dir)    
    
    def get_vocab_size(self):
        print(len(self.words))
        return len(self.words)   
 
    def to_words(self,content, words):
        """将id表示的内容转换为文字"""
        return ''.join(words[x] for x in content)
    
    
    def do_file(self,filename,  max_length=600):
        """将文件转换为id表示"""
        contents, labels = self.load_file(filename)
    
        print("begin process file 1... %s " % filename)
        data_id, label_id = [], []
        for i in range(len(contents)):
            data_id.append([self.word_to_id[x] for x in contents[i] if x in self.word_to_id])
            label_id.append(self.cat_to_id[labels[i]])
    
        # 使用keras提供的pad_sequences来将文本pad为固定长度
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
        y_pad = kr.utils.to_categorical(label_id, num_classes=len(self.cat_to_id))  # 将标签转换为one-hot表示
    
        return x_pad, y_pad
    
    
    def batch_iter(self,x, y, batch_size=256):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1
    
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]
    
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
