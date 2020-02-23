package hz.ai.help; 

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.avcodec;

import org.bytedeco.javacpp.opencv_core.Mat;

import java.awt.image.BufferedImage;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.Buffer;

import java.nio.ByteBuffer;

import javax.imageio.ImageIO;
 
import org.bytedeco.javacpp.opencv_core; 
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacpp.avutil;
 
import org.bytedeco.javacpp.opencv_imgcodecs;
 

public class VideoTest {
	
	
	
	public static void fetchFrame(String videofile, String framefile)
            throws Exception, IOException {
        FFmpegFrameGrabber ff = new FFmpegFrameGrabber(videofile); 
        ff.start();
        ff.getAudioChannels();
        String rotate =ff.getVideoMetadata("rotate");//视频的旋转角度
        int lenght = ff.getLengthInFrames();
        int i = 0;
        Frame f = null;
        Socket socket = new Socket("192.168.0.103", 6606);
        
        OutputStream  os=  socket.getOutputStream() ;
        System.out.println("begin ");
        while (i < lenght) {
            // 过滤前5帧，避免出现全黑的图片，依自己情况而定
            f = ff.grabFrame();
            
            System.out.println("ffff ");
            System.out.println("ffff "+f.image);
            if ((i > 45) && (f.image != null)) {
            	//break;
            }
            
            
            //os.write(f.image.);
            if(f.image != null){
            	
            	Java2DFrameConverter java2dFrameConverter = new Java2DFrameConverter(); 
            	BufferedImage bufferedImage= java2dFrameConverter.convert(f); 
            	
            	int size=bufferedImage.getData().getDataBuffer().getSize();
            	//System.out.println(bufferedImage.getData().getDataBuffer().);
            	ByteArrayOutputStream out = new ByteArrayOutputStream();
                ImageIO.write(bufferedImage, "jpg", out);
                size=out.size();
              
                //size=size/8;
            	//size=66897;
            	System.out.println("ssss "+size);
            	String ss="00000000000000000"+size;
            	ss=ss.substring(ss.length()-16, ss.length());
            	System.out.println("ssss "+ss);
            	os.write(ss.getBytes());
            	os.write(out.toByteArray());
            	  
            	 
            	//ImageIO.write(bufferedImage, "jpg", os);
            	os.flush();
             
            	System.out.println("send   xxxxxxxxxxxxxxxxxx");
            	//break;
            	//os.close();
            
             
            }else{
            	continue;
            }
                  
            IplImage src = null;
                
            //doExecuteFrame(f, framefile);
            i++;
        }
        
        os.close();
        
    }
    
	
    
    public static IplImage rotate(IplImage src,int angle) {
        IplImage img = IplImage.create(src.height(),src.width(),src.depth(),src.nChannels());
        opencv_core.cvTranspose(src,img);
        opencv_core.cvFlip(img,img,angle);
        return img;
    }
    
    
    public static void doExecuteFrame(Frame f,String targetFileName) {
        if (null ==f ||null ==f.image) {
        	System.out.println("return.......");
            return;
        }
        Java2DFrameConverter converter =new Java2DFrameConverter();
        BufferedImage bi =converter.getBufferedImage(f);
        
        File output =new File(targetFileName);
        try {
            ImageIO.write(bi,"jpg",output);
        }catch (IOException e) {
        	System.out.print("write-IOException   "+e.getMessage());
            e.printStackTrace();
        }
    }

 
    public static void main(String[] args){
    	try{
    		fetchFrame("E:/ai/mai/live/road.mp4","E:/ai/mai/live/oo_test1.jpg");
    		System.out.print("XXXXX");
    	}catch(IOException e){
    		System.out.print("IOException   "+e.getMessage());
    		e.printStackTrace();
    	}
    }
}
