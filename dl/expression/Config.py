class RunConfig(object):
    #train_data_dir='/root/mt/tor/dk1/train/'
    train_data_dir='./dataset/'
    dev_sample_percentage=0.01
    img_height=48
    img_width=48
    img_channels=1
    dropout_keep_prob=0.7
    learning_rate=0.001
    batch_size=16
    num_epochs=200
    evaluate_every=50
    checkpoint_every=50
    num_checkpoints=5
    device_name='/cpu:0'
    allow_soft_placement=True
    log_device_placement=False

