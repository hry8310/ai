import socket
import time
import cv2
import numpy
import video_det as md
from concurrent.futures import ThreadPoolExecutor
import threading
import functools 

pool = ThreadPoolExecutor(max_workers=2)

def rel(ft,o):
    time.sleep(3)
    print(ft.done())
    o.release()
 
def start():
    #IP地址'0.0.0.0'为等待客户端连接
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
                return buf 
            buf += newbuf
            count -= len(newbuf)
        return buf
        
    #接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，可以用来接收和发送数据。addr是连接客户端的地址。
    #没有连接则等待有连接
    while True:   
        conn, addr = s.accept()
        times=0;
        dist=[]
        out=None
        fut=None
        print('connect from:'+str(addr))
        while 1:
            try:
                start = time.time()#用于计算帧率信息
                length = recvpack(conn,16)
                print("body_length")
                if int(length)==0:
                    continue        
                stringData = recvpack(conn, int(length))#根据获得的文件长度，获取图片文件
                data = numpy.frombuffer(stringData, numpy.uint8)#将获取到的字符流数据转换成1维数组
                decimg=cv2.imdecode(data,cv2.IMREAD_COLOR)#将数组解码成图像
                if out is None:
                    print(decimg.shape)
                    out = cv2.VideoWriter('./outputs/oo_test4.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),16.0,(decimg.shape[1],decimg.shape[0]) )
                #md.det(decimg,out)
                fut=pool.submit(md.det,decimg,out) 
                end = time.time()
                seconds = end - start
                fps  = 1/seconds;
                k = cv2.waitKey(10)&0xff
                if k == 27:
                    break
            except :
                print('')
                break
        if not fut is  None:
            print('put_close....') 
            fut.add_done_callback(functools.partial(rel,o=out))
    s.close()

if __name__ == '__main__':
    start()
