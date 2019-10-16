from lib.config import cfg
import tensorflow as tf
#saver = tf.train.import_meta_graph("./logs/yolov3_loss=8876.0342.ckpt-3.meta")
saver = tf.train.import_meta_graph(cfg.Te_weight_file+".meta")
sess = tf.Session()
saver.restore(sess, cfg.Te_weight_file)
graph = sess.graph
print([node.name for node in graph.as_graph_def().node])
