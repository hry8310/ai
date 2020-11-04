package cn.hz.andtf;


import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.Vector;

public class CNN {

    //MODEL PATH
    private static final String MODEL_FILE  = "file:///android_asset/dvc.pb";
    //tensor name
    private static final String netInName  ="input_x:0";
    private static final String[] netOutName =new String[]{"prediction/y_pred:0"};

    private AssetManager assetManager;
    private TensorFlowInferenceInterface inferenceInterface;
    public Long type=10l;
    CNN(AssetManager mgr){
        assetManager=mgr;
        loadModel();
    }
    private boolean loadModel() {
        //AssetManager
        try {
            inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
            Log.d("net","[*]load model success");
        }catch(Exception e){
            Log.e("net","[*]load model failed"+e);
            return false;
        }
        return true;
    }


    private  Long runNet(Bitmap bitmap){
        int w=bitmap.getWidth();
        int h=bitmap.getHeight();

        float[] imgIn=Utils.normalizeImage(bitmap);

        inferenceInterface.feed(netInName,imgIn,1,w,h,1);
        float[] prop=new float[1];
        prop[0]=0.1f;
        inferenceInterface.feed("dropout_keep_prob",prop,1);
        inferenceInterface.run(netOutName,false);

        long[] out=new long[1];
        out[0]=10;
        inferenceInterface.fetch(netOutName[0],out);

        return out[0];
    }





    public void pred(Bitmap bitmap ) {

        Bitmap bm=Utils.bitmapResize(bitmap,224,224);

        type=runNet(bm);



    }
}
