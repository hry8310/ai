import socket
import time
import cv2
import numpy as np
from pred_net_img import YoloTest
import json
 
def start():
    address = ('0.0.0.0', 6606)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(address)
    s.listen(1)
    yolo=YoloTest()
 
    def recvpack(sock, count):
        buf = b'' 
        _len=0
        while count:
            newbuf = None
            try:
                newbuf=sock.recv(count)
            except :
                print('')
            
            if not newbuf:
                print(len(buf))
                return buf 
            buf += newbuf
            count -= len(newbuf)
        return buf
    
    def sendpack(sock,data):
    	  #sock.sendall(bytes(data,"UTF-8"))    
    	  sock.sendall(data)   
    	  print('send over')
        
    while True:   
        conn, addr = s.accept()
        times=0;
        dist=[]
         
        try :
            while 1:
                print('connect from:'+str(addr))
                start = time.time()
                length = recvpack(conn,16)
                print("body_length")
                print(int(length))
        
                stringData = recvpack(conn, int(length))
                print(stringData)
                data = np.frombuffer(stringData, np.uint8)
                decimg=cv2.imdecode(data,cv2.IMREAD_COLOR)
                print("decimg")
                print(data)
                cv2.imwrite("./test/test.jpg",decimg)
                imgg=yolo.get_img(decimg)
                _,img_np=cv2.imencode('.jpg',imgg)
                bff=img_np.tostring()
                blen=len(bff)
                blen="0000000000000000"+str(blen)
                blen=blen[len(blen)-16:len(blen)]
                print(blen)
                sendpack(conn,bytes(blen,"UTF-8"))
                end = time.time()
                sendpack(conn,bff)
                seconds = end - start
                fps  = 1/seconds;
                k = cv2.waitKey(10)&0xff
                if k == 27:
                    break
        except Exception as r:
             print(' %s' %(r))
             
    s.close() 
if __name__ == '__main__':
    start()
