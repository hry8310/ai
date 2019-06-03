import tensorflow as tf
class Config(object):
    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 10000  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

def w_variable( shape ,stddev=0.1 ,name='weights'):
    return tf.get_variable(name,
        shape=shape,
        dtype = tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32))

def b_variable(shape ,init=0.1,name='biases'):
    return tf.get_variable(name,
        shape=shape,
        initializer=tf.constant_initializer(init))

def h_conv(input,weights,biases,stride=1,padding='SAME' ,name='covn1'):
    conv=tf.nn.conv1d(input, weights, stride=stride, padding=padding)
    pre_activation = tf.nn.bias_add(conv, biases)
    out = tf.nn.relu(pre_activation, name= name)
    return out

def h_max(convv ,  ksize=[1,3,3,1],strides=[1,2,2,1], padding='SAME', name='pool1'):
    pool=tf.nn.pool( convv,window_shape=ksize,strides=strides,pooling_type='AVG',padding=padding,name=name)
    return pool

def h_fc(lay, weights,biases, act_fun=None , name='fc1'):
    if act_fun is None:
        return tf.add(tf.matmul( lay , weights ),biases,name=name)
    else:
        layy=tf.matmul( lay , weights )+biases
        return act_fun(layy,name=name)

class TextCNN(object):
    def __init__(self,config):
        self.config=config
        self.input_x=tf.placeholder(tf.int32,shape=[None, self.config.seq_length] ,name='input_x')
        self.input_y=tf.placeholder(tf.float32,shape=[None, self.config.num_classes] ,name='input_y')
        self.keep_prob=tf.placeholder(tf.float32, name='keep_prob')

        self.cnn() 


    def cnn(self):
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('cov1'):
            w=w_variable(shape=[self.config.kernel_size,self.config.batch_size, self.config.num_filters],stddev=0.1)
            b=b_variable(shape=[ self.config.num_filters],init=0.1)
            conv1=h_conv( embedding_inputs ,w , b ,name='conv1')

        with tf.name_scope('pool1'):
            norm=h_max(conv1,ksize=[3] ,strides=[2],name='pooll')

        with tf.name_scope('fc1'):
            num_total_unit = norm.get_shape()[1:3].num_elements()
            reshape=tf.reshape(norm,shape=[-1,num_total_unit])
            weights=w_variable(shape=[num_total_unit,128],stddev=0.005,name='w2')
            biases=b_variable(shape=[128],init=0.1,name='b2')
            local3 = h_fc(reshape, weights, biases,act_fun=tf.nn.relu, name='fc1')
      
        with tf.name_scope('dropout'): 
            dropout=tf.nn.dropout(local3,keep_prob=0.7,name='dropout')

        with tf.name_scope("score"):
            weights=w_variable(shape=[128,self.config.num_classes],stddev=0.005,name='w3')
            biases=b_variable(shape=[self.config.num_classes],init=0.1,name='b3')
            self.logits = h_fc(dropout, weights, biases, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
  
        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            grads_and_vars = optimizer.compute_gradients(self.loss)
            self.optim = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
   
            



