
import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import multiprocessing as mp


return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
#video_path      = "./docs/images/road.mp4"
# video_path      = 0
video_path      = "http://admin:admin@192.168.1.105:8081"
num_classes     = 80
input_size      = 416

new_img=None
def read_img(q):
    vid = cv2.VideoCapture(video_path)
    while True:
        fnum=vid.get(cv2.CAP_PROP_FRAME_COUNT)
        print("read..........")              
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if q.empty() ==True:
                print('is not full.......')
                q.put(frame)     
            else:
                print("is full...")                       
        else:
            raise ValueError("No image!")
        
         
        
def view(q): 
    graph           = tf.Graph()
    return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)
    with tf.Session(graph=graph) as sess:
         
        while True:
            frame =q.get()        	
                   
            image = Image.fromarray(frame)
            print("img.......")
            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]
            prev_time = time.time()
    
            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})
    
            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
    
            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            image = utils.draw_bbox(frame, bboxes)
    
            curr_time = time.time()
            exec_time = curr_time - prev_time
            result = np.asarray(image)
            info = "time: %.2f ms" %(1000*exec_time)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break        
            
def main():
    processes = []
    qu=mp.Queue(1)
     
        
    
    processes.append(mp.Process(target=read_img,args=(qu,)))
    processes.append(mp.Process(target=view,args=(qu,)))
    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()
        
if __name__ == '__main__':
    main()


