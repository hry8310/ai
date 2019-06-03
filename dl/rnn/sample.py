import tensorflow as tf
from utils import TextConverter
from model import RNN
from config import Config
import os
import codecs


config = Config()

def main(_):
    mode_path = os.path.join('log', config.name)
    with codecs.open(config.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text, config.max_vocab)


    model = RNN(converter.vocab_size, training=False,
                    rnn_size=config.rnn_size, num_layers=config.num_layers,
                    use_embedding=config.use_embedding,
                    embedding_size=config.embedding_size)
    
    cp=os.path.join(mode_path,'model-60')
    model.load(cp)

    start = converter.text_to_arr('秋')
    arr = model.sample(30, start, converter.vocab_size)
    print(converter.arr_to_text(arr))


if __name__ == '__main__':
    tf.app.run()
