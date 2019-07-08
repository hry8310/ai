import socket
import time
import cv2
import numpy
import model as md

 
def start():
    address = ('0.0.0.0', 6606)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(address)
    s.listen(1)
 
    def recvpack(sock, count):
        buf = b''#buf是一个byte类型
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
        
    while True:   
        conn, addr = s.accept()
        times=0;
        dist=[]
        out=None
        try :
            while 1:
                print('connect from:'+str(addr))
                start = time.time()
                length = recvpack(conn,16)
                print("body_length")
                print(int(length))
        
                stringData = recvpack(conn, int(length))
                data = numpy.frombuffer(stringData, numpy.uint8)
                decimg=cv2.imdecode(data,cv2.IMREAD_COLOR)
                if out is None:
                    print(decimg.shape)
                    out = cv2.VideoWriter('./outputs/oo_test4.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),16.0,(decimg.shape[1],decimg.shape[0]) )
                times,dist=md.moving(decimg,out,times,dist)
        
 
                end = time.time()
                seconds = end - start
                fps  = 1/seconds;
                k = cv2.waitKey(10)&0xff
                if k == 27:
                    break
        except :
            print() 
        out.release()     
    s.close() 
if __name__ == '__main__':
    start()
