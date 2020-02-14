import tensorflow as tf
saver = tf.train.import_meta_graph("./log/ck/checkpoints/model-8200.meta")
#saver = tf.train.import_meta_graph(cfg.Te_weight_file+".meta")
sess = tf.Session()
saver.restore(sess, "./log/ck/checkpoints/model-8200")
graph = sess.graph
print([node.name for node in graph.as_graph_def().node])
