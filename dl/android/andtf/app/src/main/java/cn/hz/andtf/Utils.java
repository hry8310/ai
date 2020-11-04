package cn.hz.andtf;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.Rect;
import android.util.Log;

import java.util.Vector;

public class Utils {

    public static Bitmap copyBitmap(Bitmap bitmap){
        return bitmap.copy(bitmap.getConfig(),true);
    }

    public static  float[] normalizeImage(Bitmap bitmap){
        int w=bitmap.getWidth();
        int h=bitmap.getHeight();
        float[] floatValues=new float[w*h];
        int[]   intValues=new int[w*h];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        float imageMean=127.5f;
        float imageStd=128;

        for (int i=0;i<intValues.length;i++){
            final int val=intValues[i];

            floatValues[i  + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd
                    + (((val >> 8) & 0xFF) - imageMean) / imageStd
                    + ((val & 0xFF) - imageMean) / imageStd;
        }
        return floatValues;
    }

    public static  Bitmap bitmapResize(Bitmap bm, int nw, int nh) {
        int width = bm.getWidth();
        int height = bm.getHeight();

        Matrix matrix = new Matrix();
        // RESIZE THE BIT MAP
        matrix.postScale(((float) nw) / width, ((float) nh) / height);
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, true);
        return resizedBitmap;
    }
}
