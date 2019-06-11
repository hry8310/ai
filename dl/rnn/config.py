class Config(object):
    name='default'
    num_seqs=80
    num_steps=100
    rnn_size=120 
    num_layers=2
    use_embedding=False
    embedding_size=128
    learning_rate=0.01
    train_keep_prob=0.5
    input_file='data/poetry.txt'
    max_steps=1000
    save_every_n=1000
    log_every_n=10
    max_vocab=3500 

   
