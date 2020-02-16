import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from model import CNN
import matplotlib.pyplot as plt
import utils 
from Config import RunConfig



config=RunConfig()

def get_test_file(file_dir):
    pics = []
    for file in os.listdir(file_dir):
        name = file
        pics.append(file_dir + file)

    print('There are %d ' %(len(pics)))
    image_list = np.hstack((pics))
    np.random.shuffle(image_list)
    print(image_list)

    return image_list


def get_one_image(train):
   imgArr=[]
   global imageForShow
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   image = Image.open(img_dir)
   imageForShow = image
   # plt.imshow(image)
   # plt.show()
   image = image.resize([224, 224])
   image = np.array(image)
   image = utils.rgb2gray(image)
   imgArr.append(image)
   imgArr=np.array(imgArr)
   return imgArr


def evaluate_one_image():
   DIR_PRE = os.getcwd() + '/'
   test_dir = DIR_PRE + 'data/test1/'
   test = get_test_file(test_dir)
   image_array = get_one_image(test)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 2

       image = tf.cast(image_array, tf.float32)
       image = tf.reshape(image, [1,224, 224, 1])
       cnn = CNN(
            n_classes=2,
            img_height=config.img_height,
            img_width=config.img_width,
            img_channel=config.img_channels,
            device_name=config.device_name
           )
       x = tf.placeholder(tf.float32, shape=[1,224, 224, 1])

       DIR_PRE = os.getcwd() + '/'
       logs_train_dir = DIR_PRE + 'log/ck/checkpoints'
       checkpoint_prefix = os.path.join(logs_train_dir, 'model-550')
       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           saver.restore(sess, checkpoint_prefix)
           print('No checkpoint file found')

           prediction = sess.run(cnn.h_pool_3_flat, feed_dict={cnn.input_x: image_array,cnn.dropout_keep_prob:1.0})
           cl0=np.mean(np.square(prediction-utils.get_np_arr(0)))
           cl1=np.mean(np.square(prediction-utils.get_np_arr(1)))
           print(prediction)
           print('prediction  %f',prediction)
           print(cl0)
           print(cl1)
           max_index = np.argmax(prediction)
           if cl0 > cl1:
               print('This is a cat with possibility')
               plt.title("This is a cat")
           else:
               print('This is a dog with possibility')
               plt.title("This is a dog")
           

evaluate_one_image()

