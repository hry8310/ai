package cn.hz.tordc;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.provider.MediaStore;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    CNN cnn;
    ImageView imageView;
    TextView textView;
    Bitmap imageBitmap =null;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        initMain();
        imageView.setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View view){
/*
                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

                startActivityForResult(cameraIntent,cameraRequestCode);
*/
                Intent intent= new Intent(Intent.ACTION_PICK,null);
                intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,"image/*");
                startActivityForResult(intent, 0x1);

            }


        });

    }

    private void initMain(){
        cnn = new CNN(Utils.assetFilePath(this, "dvc.pt"));

        imageView = findViewById(R.id.imageView);
        textView=(TextView)findViewById(R.id.predRes);
        imageBitmap=readAssetsImg("dog.jpg");
        pred();
    }

    private  Bitmap readAssetsImg(String filename){
        Bitmap bitmap;
        AssetManager asm=getAssets();
        try {
            InputStream is=asm.open(filename);
            bitmap= BitmapFactory.decodeStream(is);
            is.close();
        } catch (IOException e) {

            e.printStackTrace();
            return null;
        }
        return Utils.copyBitmap(bitmap); //返回mutable的image
    }

    private  void pred(){
        String pred = cnn.predict(imageBitmap);
        textView.setText("类型：  "+pred);
        imageView.setImageBitmap(imageBitmap);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){


        try {
            imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());
        }catch (Exception e){
            return;
        }
        pred();
        super.onActivityResult(requestCode, resultCode, data);
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}
