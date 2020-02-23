package hz.ai.help;
 
 
 
 
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import  java.math.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Random;

import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.tensorflow.*;
@SuppressWarnings("unused")
public class Test {

    private static final Integer ONE = 1;

    public static void main1(String[] args) {
       
        int input [][] =new int[1][600];
         
        Graph graph = new Graph();
        SavedModelBundle b = SavedModelBundle.load("E:/ai/mai/yolo/成功版本/119/yolo3_9/", "test");
        Session tfSession = b.session();
        Operation operationPredict = b.graph().operation("predict");   //要执行的op
        Output output = new Output(operationPredict, 0);
        Tensor input_X = Tensor.create(input);
        Tensor out= tfSession.runner().feed("input_x",input_X).fetch(output).run().get(0);
        System.out.println(out);
        float [][] ans = new float[1][10];
        out.copyTo(ans);
        float M=0;
        int index1=0;
        index1 =getMax(ans[0]);
        System.out.println(index1);
        System.out.println("------"); 

        //System.out.println(mp.get(getMax(ans[1])));
    }
    
    
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
      // Mat y1 = opencv_imgcodecs.imread("y_one.jpg",0);
    	Mat y1 = opencv_imgcodecs.imread("y_one.jpg",0);
    	
    	y1.getByteBuffer();
     
       //y1.resi
    	try(Graph graph = new Graph()){
            byte[] graphBytes = org.apache.commons.io.IOUtils.toByteArray(new java.io.FileInputStream("E:/ai/mai/yolo/成功版本/119/yolo3_9//test.pb"));
            graph.importGraphDef(graphBytes);
            int flen=1*200*200*3;
            float[] ff=new float[1*416*416*3];
            FloatBuffer f=floatToBuffer(ff) ;
            long[] size={1l,416l,416l,3l};
            
            Tensor inp=Tensor.create(size,y1.getByteBuffer().asFloatBuffer());
            try(Session session = new Session(graph)){
                List<Tensor<?>> outs = session.runner()
                        .feed("input_data_1",inp).feed("cfg_input/training",Tensor.create(true))
                        .fetch("pred_sbbox/concat").fetch("pred_mbbox/concat").fetch("pred_lbbox/concat").run();
                Tensor<?> out=outs.get(0);
                
                float[] r = new float[1];
                //out.copyTo(r);
                System.out.println(out);
            }
        }
        //System.out.println(mp.get(getMax(ans[1])));
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
