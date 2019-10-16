
from lib.config import cfg
import tensorflow as tf

def conv(input_data, filters_shape, trainable, name, downsample=False, act=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if act == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv



def residual(input_data, input_channel, filter_num1, filter_num2, trainable, name):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = conv(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = conv(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output



def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output

def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output



def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = conv(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = conv(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = residual(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = conv(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = residual(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        input_data = conv(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        for i in range(8):
            input_data = residual(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = conv(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = residual(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = conv(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = residual(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data


def appendnet(input_data,route_1, route_2 ,trainable,num_class):
    input_data = conv(input_data, (1, 1, 1024,  512), trainable, 'conv52')
    input_data = conv(input_data, (3, 3,  512, 1024), trainable, 'conv53')
    input_data = conv(input_data, (1, 1, 1024,  512), trainable, 'conv54')
    input_data = conv(input_data, (3, 3,  512, 1024), trainable, 'conv55')
    input_data = conv(input_data, (1, 1, 1024,  512), trainable, 'conv56')

    conv_lobj_branch = conv(input_data, (3, 3, 512, 1024), trainable, name='conv_lobj_branch')
    conv_lbbox = conv(conv_lobj_branch, (1, 1, 1024, 3*(num_class + 5)),
                                          trainable=trainable, name='conv_lbbox', act=False, bn=False)

    input_data = conv(input_data, (1, 1,  512,  256), trainable, 'conv57')
    input_data = upsample(input_data, name='upsample0', method=cfg.Upsample_method)
        
    with tf.variable_scope('route_1'):
        input_data = tf.concat([input_data, route_2], axis=-1)
        
    input_data = conv(input_data, (1, 1, 768, 256), trainable, 'conv58')
    input_data = conv(input_data, (3, 3, 256, 512), trainable, 'conv59')
    input_data = conv(input_data, (1, 1, 512, 256), trainable, 'conv60')
    input_data = conv(input_data, (3, 3, 256, 512), trainable, 'conv61')
    input_data = conv(input_data, (1, 1, 512, 256), trainable, 'conv62')
                                         

    conv_mobj_branch = conv(input_data, (3, 3, 256, 512),  trainable, name='conv_mobj_branch' )
    conv_mbbox = conv(conv_mobj_branch, (1, 1, 512, 3*(num_class + 5)),
                                          trainable=trainable, name='conv_mbbox', act=False, bn=False)

    input_data = conv(input_data, (1, 1, 256, 128), trainable, 'conv63')
    input_data = upsample(input_data, name='upsample1', method=cfg.Upsample_method)

    with tf.variable_scope('route_2'):
        input_data = tf.concat([input_data, route_1], axis=-1)

    input_data = conv(input_data, (1, 1, 384, 128), trainable, 'conv64')
    input_data = conv(input_data, (3, 3, 128, 256), trainable, 'conv65')
    input_data = conv(input_data, (1, 1, 256, 128), trainable, 'conv66')
    input_data = conv(input_data, (3, 3, 128, 256), trainable, 'conv67')
    input_data = conv(input_data, (1, 1, 256, 128), trainable, 'conv68')

    conv_sobj_branch = conv(input_data, (3, 3, 128, 256), trainable, name='conv_sobj_branch')
    conv_sbbox = conv(conv_sobj_branch, (1, 1, 256, 3*(num_class + 5)),
                                          trainable=trainable, name='conv_sbbox', act=False, bn=False)

    return conv_lbbox, conv_mbbox, conv_sbbox

def yolonet(input_data,trainable,num_class):
    route_1, route_2, input_data = darknet53(input_data, trainable)
    lbox, mbox, sbox=appendnet(input_data,route_1, route_2 ,trainable, num_class)
    return lbox, mbox, sbox



