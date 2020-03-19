package hz.ai.help;
 
 
 
 

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

import javax.imageio.ImageIO;
 

 
import org.opencv.videoio.VideoCapture; 
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.bytedeco.javacpp.opencv_imgcodecs; 

import org.tensorflow.*;
@SuppressWarnings("unused")
public class CamTest {
 
    
    public static FloatBuffer floatToBuffer(float[]a){
    	ByteBuffer mbb=ByteBuffer.allocateDirect(a.length*4);
    	mbb.order(ByteOrder.nativeOrder());
    	FloatBuffer mBuffer=mbb.asFloatBuffer();
    	mBuffer.put(a);
    	mBuffer.position(0);
    	return mBuffer;
    }
    
     
    	 
    public static void main(String[] args) throws IOException ,Exception {
    	 System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    	 VideoCapture camera=new VideoCapture("http://admin:admin@192.168.1.113:8081" );

        if(!camera.isOpened()){
 			System.out.println("Camera Error");
 			return ;
 		 }
 		Mat frame=new Mat();
		while(true){
			camera.read(frame);
            Mat src =new Mat(frame.rows(),frame.cols(),frame.type());

             if(src.dataAddr()==0){
                  System.out.println("打开文件出错 "+src);
             }
         
             Mat dst = new Mat();
             Imgproc.resize(src, dst, new Size(224, 224), 0, 0, Imgproc.INTER_CUBIC);
         
             int img_rows = dst.rows();
         // 图像列:宽度width
             int img_colums = dst.cols();
         // 图像通道:维度dims/channels
             int img_channels = dst.channels();
             float[]  bsf=new float[224*224*1];
         
            for(int j=0;j<img_rows;j++){
                for(int k=0; k<img_colums;k++){
               
                float p=(float)(( (float)(dst.get(j,k)[0]*0.299 )
     				   +(float)(dst.get(j,k)[1]*0.587 )
     				   +(float)(dst.get(j,k)[2]*0.114 )
     				 )/255.0);
                bsf[j*224+k]=p;
             }
         }

        
		  
		 try(Graph graph = new Graph()){
	            byte[] graphBytes = org.apache.commons.io.IOUtils.toByteArray(new java.io.FileInputStream("E:/ai/mai/ai/dl/dvc/test.pb"));
	            graph.importGraphDef(graphBytes);
	            FloatBuffer f=floatToBuffer(bsf) ;
	            System.out.println(f.get(224*224-12));
	            long[] size={1l,224l,224l,1l};
	            
	            
	            Tensor inp=Tensor.create(size,f);
	            try(Session session = new Session(graph)){
	                List<Tensor<?>> outs = session.runner()
	                        .feed("input_x:0",inp).feed("dropout_keep_prob",Tensor.create(1.0f))
	                        .fetch("prediction/y_pred:0") .run();
	                Tensor<?> out=outs.get(0);
	                
	                long[] r = new long[1];
	                r[0]=10;
	                out.copyTo(r);
	                if(r[0]==0){
	                	System.out.println("this is dog");
	                }else{
	                	System.out.println("this is cat");
		                
	                }
	            }
	        }
		}
         
         
         
      
         
         
    }

    public static int getMax(float[] a){
        float M=0;
        int index2=0;
        for(int i=0;i<10;i++){
            if(a[i]>M){
                M=a[i];
                index2=i;
            }
        }
        return index2;
    }

}
