import numpy as np
import copy
import time
import tensorflow as tf
import collections

import pickle


def batch_generator(arr, n_seqs, n_steps):
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    while True:
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y


class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        counter = collections.Counter(text)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.vocab, _ = zip(*count_pairs)
        self.vocab2 = dict(zip(self.vocab, range(len(self.vocab))))

        #self.word_to_int_table = {c: i for i, c in enumerate(self.vocab2)}
        self.word_to_int_table = np.copy(self.vocab2)
        self.int_to_word_table = dict(enumerate(self.vocab2))
        #print(self.vocab)
        #print(self.int_to_word_table)
    


    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')


    def text_to_arr0(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        narr=np.array(arr)
        return narr

    def text_to_arr(self, text):
        arr=np.array(list(map(self.vocab2.get, text)))
        return arr

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)
