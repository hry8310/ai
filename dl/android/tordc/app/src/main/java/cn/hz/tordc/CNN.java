package cn.hz.tordc;

import android.graphics.Bitmap;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;


public class CNN {

    Module model;
    float[] mean = {0.485f, 0.456f, 0.406f};
    float[] std = {0.229f, 0.224f, 0.225f};

    public CNN(String modelPath){

        model = Module.load(modelPath);

    }



    public Tensor preprocess(Bitmap bitmap, int size){

        bitmap = Bitmap.createScaledBitmap(bitmap,size,size,false);
        return TensorImageUtils.bitmapToFloat32Tensor(bitmap,this.mean,this.std);

    }

    public int argMax(float[] inputs){

        int maxIndex = -1;
        float maxvalue = 0.0f;

        for (int i = 0; i < inputs.length; i++){

            if(inputs[i] > maxvalue) {

                maxIndex = i;
                maxvalue = inputs[i];
            }

        }


        return maxIndex;
    }

    public String predict(Bitmap bitmap){

        Tensor tensor = preprocess(bitmap,200);

        IValue inputs = IValue.from(tensor);
        Tensor outputs = model.forward(inputs).toTensor();
        float[] scores = outputs.getDataAsFloatArray();

        int classIndex = argMax(scores);
        String ty="猫";
        if(classIndex==1){
            return "狗";
        }
        return ty;
       // return Constants.IMAGENET_CLASSES[classIndex];

    }

}

