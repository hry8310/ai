import tensorflow as tf
from utils import TextConverter, batch_generator
from model import RNN
import os
import codecs

from config import Config


config=Config()

def main(_):
    model_path = os.path.join('log', config.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    sum_path = os.path.join('summ', config.name)
    if os.path.exists(sum_path) is False:
        os.makedirs(sum_path)
    with codecs.open(config.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text, config.max_vocab)
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text)
    g = batch_generator(arr, config.num_seqs, config.num_steps)
    print(converter.vocab_size)
    model = RNN(converter.vocab_size,
                    num_seqs=config.num_seqs,
                    num_steps=config.num_steps,
                    rnn_size=config.rnn_size,
                    num_layers=config.num_layers,
                    learning_rate=config.learning_rate,
                    train_keep_prob=config.train_keep_prob,
                    use_embedding=config.use_embedding,
                    embedding_size=config.embedding_size
                    )
    model.train(g,
                sum_path,
                config.max_steps,
                
                model_path,
                config.save_every_n,
                config.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()
