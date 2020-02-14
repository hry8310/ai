import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tqdm import tqdm

output_node_names = ["input_x",'dropout_keep_prob', "prediction/y_pred"]
def export_model(input_checkpoint, output_graph):
    #这个可以加载saver的模型
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
        sess=sess,
        input_graph_def=input_graph_def,# 等于:sess.graph_def
        output_node_names=output_node_names )# 如果有多个输出节点，以逗号隔开这个是重点，输入和输出的参数都需要在这里记录

        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出

export_model('./log/ck/checkpoints/model-8200','test.pb')
#export_model('./logs/yolov3_coco_demo.ckpt','test.pb')

