package hz.ai.help;
 
 
 
 

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import org.tensorflow.*;
@SuppressWarnings("unused")
public class Test310 {

    private static final Integer ONE = 1;
 
    
    public static FloatBuffer floatToBuffer(float[]a){
    	ByteBuffer mbb=ByteBuffer.allocateDirect(a.length*4);
    	mbb.order(ByteOrder.nativeOrder());
    	FloatBuffer mBuffer=mbb.asFloatBuffer();
    	mBuffer.put(a);
    	mBuffer.position(0);
    	return mBuffer;
    }
    
    public static FloatBuffer byteToBuffer(float[]a){
    	ByteBuffer mbb=ByteBuffer.allocateDirect(a.length*4);
    	mbb.order(ByteOrder.nativeOrder());
    	FloatBuffer mBuffer=mbb.asFloatBuffer();
    	mBuffer.put(a);
    	mBuffer.position(0);
    	return mBuffer;
    }
    	 
    public static void main(String[] args) throws Exception {
    	 System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
         //将文件读入为OpenCV的Mat格式。注意测试时，路径不要包括中文
         Mat src = Imgcodecs.imread("E:/ai/a.jpg");
         
         if(src.dataAddr()==0){
             System.out.println("打开文件出错 "+src);
         }
         System.out.println( src);
         
         Mat dst = new Mat();
         Imgproc.resize(src, dst, new Size(416, 416), 0, 0, Imgproc.INTER_AREA);
         System.out.println( dst.channels());
         Mat dst2 = new Mat();
         Imgproc.cvtColor(src, dst2,Imgproc.COLOR_BGR2GRAY);
         
         System.out.println( dst2.channels()); 
         
         float[]  bsf=new float[416*416*3];
         for(int i=0;i<416;i++){
        	 for(int j=0;j<416;j++){
        		 byte[] d=new byte[3];
        		 dst.get(i, j, d);
        		 bsf[i*416+j]=Float.valueOf(d[0]);
        		 bsf[416*416+i*416+j]=Float.valueOf(d[1]);
        		 bsf[2*416*416+i*416+j]=Float.valueOf(d[2]);
        	 }
         }
         
         byte[] d=new byte[3];
		 dst.get(4, 1, d);
		 System.out.println( d[0]);

		 System.out.println( Float.valueOf(d[0]));
		 
		 try(Graph graph = new Graph()){
	            byte[] graphBytes = org.apache.commons.io.IOUtils.toByteArray(new java.io.FileInputStream("E:/ai/mai/yolo/成功版本/119/yolo3_9//test.pb"));
	            graph.importGraphDef(graphBytes);
	            int flen=1*200*200*3;
	            float[] ff=new float[1*416*416*3];
	            FloatBuffer f=floatToBuffer(bsf) ;
	            long[] size={1l,416l,416l,3l};
	            
	            Tensor inp=Tensor.create(size,f);
	            try(Session session = new Session(graph)){
	                List<Tensor<?>> outs = session.runner()
	                        .feed("input_data_1",inp).feed("cfg_input/training",Tensor.create(true))
	                        .fetch("pred_sbbox/concat").fetch("pred_mbbox/concat").fetch("pred_lbbox/concat").run();
	                Tensor<?> out=outs.get(0);
	                
	                float[] r = new float[1];
	                //out.copyTo(r);
	                System.out.println(outs);
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
