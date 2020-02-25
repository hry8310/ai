package hz.ai.help;
 

import sun.misc.BASE64Decoder;
import sun.misc.BASE64Encoder;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import  java.math.*;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Random;

import javax.imageio.ImageIO;

import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.tensorflow.*;
@SuppressWarnings("unused")
public class YoloClientImg {

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
    	 
    public static void main0(String[] args) throws Exception {
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
    public static String getImageStr(String imgFile) {
        InputStream inputStream = null;
        byte[] data = null;
        try {
            inputStream = new FileInputStream(imgFile);
            data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        // 加密
        BASE64Encoder encoder = new BASE64Encoder();
        return encoder.encode(data);
    }
    public static void main(String[] args) throws Exception{
    	 //File output =new File("E:/ai/mai/test1.jpg");
   	     File output =new File("E:/yolo/test/test4.jpg");
    	 BufferedImage bufferedImage= ImageIO.read(output);
    	 
    	 ByteArrayOutputStream out = new ByteArrayOutputStream();
         ImageIO.write(bufferedImage, "jpg", out);
         
        //String strImg = getImageStr("E:/ai/mai/test1.jpg");
        
        Socket socket = new Socket("127.0.0.1", 6606);
        
        OutputStream  os=  socket.getOutputStream() ;
        int size=out.size();
        String ss="00000000000000000"+size;
    	ss=ss.substring(ss.length()-16, ss.length());
    	System.out.println("ssss "+ss);
    	os.write(ss.getBytes());
    	os.write(out.toByteArray());
    	InputStream ips= socket.getInputStream();
    	byte[] b0 =readLv(ips,16);
    	String bl=new String(b0);
    	System.out.println(bl);
    	int ll=Integer.valueOf(bl);
    	byte[] b =readLv(ips,ll);
    	
    	 
    	
    	ByteArrayInputStream in = new ByteArrayInputStream(b); 
    	BufferedImage bufferedImage2= ImageIO.read(in);
     
    	
    	FileOutputStream out2 = new FileOutputStream("d:/yolojpg/test2.jpg");//输出图片的地址
        ImageIO.write(bufferedImage2, "jpeg", out2);
    	
    }
    
    public static byte[] readLv(InputStream channel,int length) throws IOException {

		byte[] b=new byte[length];
		int re=0;
		while (true) {
			int l=channel.read(b,re,length-re);
			re=re+l;
			if(re>=length){
				break;
			}

			 
		}
		 
        return b;
    }
    

    public static void mainJson(String[] args) throws Exception{
    	 //File output =new File("E:/ai/mai/test1.jpg");
   	     File output =new File("E:/yolo/test/test4.jpg");
    	 BufferedImage bufferedImage= ImageIO.read(output);
    	 
    	 ByteArrayOutputStream out = new ByteArrayOutputStream();
         ImageIO.write(bufferedImage, "jpg", out);
         
        //String strImg = getImageStr("E:/ai/mai/test1.jpg");
        
        Socket socket = new Socket("127.0.0.1", 6606);
        
        OutputStream  os=  socket.getOutputStream() ;
        int size=out.size();
        String ss="00000000000000000"+size;
    	ss=ss.substring(ss.length()-16, ss.length());
    	System.out.println("ssss "+ss);
    	os.write(ss.getBytes());
    	os.write(out.toByteArray());
    	InputStream ips= socket.getInputStream();
    	String res="";
    	while(true){
    		byte[] b=new byte[2048*2]; 
    		int i=ips.read(b);
    		if(i>0){
    			System.out.println(new String(b));
    			res=new String(b);
    			break;
    		}
    	}
    	if(res.length()>0){
    		Graphics g = bufferedImage.getGraphics();
            g.setColor(Color.RED);//画笔颜色
           
    		List<List<Double>> list=JSONUtil.encode(res, List.class);
    		System.out.println(list.size());
    		for(List<Double> ls:list){
    			g.drawRect(ls.get(0).intValue(), ls.get(1).intValue(), ls.get(2).intValue()-ls.get(0).intValue(), ls.get(3).intValue()-ls.get(1).intValue());//
    			 
    		}
    	}
    	
    	FileOutputStream out2 = new FileOutputStream("d:/test2.jpg");//输出图片的地址
        ImageIO.write(bufferedImage, "jpeg", out2);
    	
    }
}
