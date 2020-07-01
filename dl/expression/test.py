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


def get_one_image(img_dir):

   imgArr=[]
   image = Image.open(img_dir)
   # plt.imshow(image)
   # plt.show()
   image = image.resize([48, 48])
   image = np.array(image)
   image = utils.rgb2gray(image)
   imgArr.append(image)
   imgArr=np.array(imgArr)
   return imgArr


def evaluate_one_image():
   DIR_PRE = os.getcwd() + '/'
   test_dir = DIR_PRE + 'dataset/jackman.png'
   image_array = get_one_image(test_dir)

   with tf.Graph().as_default():

       image = tf.cast(image_array, tf.float32)
       image = tf.reshape(image, [1,config.img_height, config.img_width, 1])
       cnn = CNN(
            n_classes=7,
            img_height=config.img_height,
            img_width=config.img_width,
            img_channel=config.img_channels,
            device_name=config.device_name
           )
       x = tf.placeholder(tf.float32, shape=[1,48, 48, 1])

       DIR_PRE = os.getcwd() + '/'
       logs_train_dir = DIR_PRE + 'log/ck/checkpoints'
       checkpoint_prefix = os.path.join(logs_train_dir, 'model-9950')
       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           saver.restore(sess, checkpoint_prefix)

           prediction = sess.run(cnn.y_pred, feed_dict={cnn.input_x: image_array,cnn.dropout_keep_prob:1.0})
           print(prediction)
           print('prediction  %f',prediction)
           max_index = np.argmax(prediction)

           if prediction==0:
               print("This is anger")

           if prediction==1:
               print("This is disgust")

           if prediction==2:
               print("This is fear")


           if prediction==3:
               print("This is happy")

           if prediction==4:
               print("This is sad")

           if prediction==5:
               print("This is surprised")

           if prediction==6:
               print("This is neutral")





evaluate_one_image()
evaluate_one_image()

