import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from model import UCNN
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
   imgArr2=[]
   global imageForShow
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   image = Image.open(img_dir)
   imageForShow = image
   # plt.imshow(image)
   # plt.show()
   image_1 = image.resize([224, 224])
   image_1 = np.array(image_1)
   image_1 = utils.rgb2gray(image_1)
   imgArr.append(image_1)
   imgArr=np.array(imgArr)
   image_2 = image.resize([112, 112])
   image_2 = np.array(image_2)
   image_2 = utils.rgb2gray(image_2)
   imgArr2.append(image_2)
   imgArr2=np.array(imgArr2)
   return imgArr,imgArr2


def evaluate_one_image():
   DIR_PRE = os.getcwd() + '/'
   test_dir = DIR_PRE + 'data/test1/'
   test = get_test_file(test_dir)
   image_array , image_array2 = get_one_image(test)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 2

       cnn = UCNN(
            n_classes=2,
            img_height=config.img_height,
            img_width=config.img_width,
            img_channel=config.img_channels,
            device_name=config.device_name
           )

       DIR_PRE = os.getcwd() + '/'
       logs_train_dir = DIR_PRE + 'log/ck/checkpoints'
       checkpoint_prefix = os.path.join(logs_train_dir, 'model-9950')
       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           saver.restore(sess, checkpoint_prefix)
           print('No checkpoint file found')

           prediction = sess.run(cnn.y_pred, feed_dict={cnn.len.input_x: image_array,
                                                        cnn.sh.input_x: image_array2,
                                                        cnn.len.dropout_keep_prob:1.0,
                                                        cnn.sh.dropout_keep_prob:1.0,
                                                        cnn.dropout_keep_prob:1.0,
                                                       }
          )
           print('prediction  %f',prediction)
           max_index = np.argmax(prediction)
           if prediction==1:
               print('This is a cat with possibility')
               plt.title("This is a cat")
           else:
               print('This is a dog with possibility')
               plt.title("This is a dog")
           plt.imshow(imageForShow)
           plt.show()

evaluate_one_image()

