package org.pytorch.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.BufferedReader;

import java.util.List;
import java.util.ArrayList;

import java.nio.FloatBuffer;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

  private static void checkOutBufferCapacity(
      FloatBuffer outBuffer, int tensorWidth, int tensorHeight) {
    if (3 * tensorWidth * tensorHeight > outBuffer.capacity()) {
      throw new IllegalStateException("Buffer underflow");
    }
  }

  public static void bitmapToFloatBuffer(
      final Bitmap bitmap,
      final int width,
      final int height,
      final FloatBuffer outBuffer) {
    checkOutBufferCapacity(outBuffer, width, height);

    final int pixelsCount = height * width;
    final int[] pixels = new int[pixelsCount];
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
    final int offset_g = pixelsCount;
    final int offset_b = 2 * pixelsCount;
    for (int i = 0; i < pixelsCount; i++) {
      final int c = pixels[i];
      float r = ((c >> 16) & 0xff);
      float g = ((c >> 8) & 0xff);
      float b = ((c) & 0xff);
      outBuffer.put(i, b);
      outBuffer.put(offset_g + i, g);
      outBuffer.put(offset_b + i, r);
    }
  }

  public static Tensor bitmapToFloat32Tensor(final Bitmap bitmap) {
    int width = bitmap.getWidth();
    int height = bitmap.getHeight();
    final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * width * height);
    bitmapToFloatBuffer(bitmap, width, height, floatBuffer);
    Log.e("floatBuffer[0]", Float.toString(floatBuffer.get(0)));
    Log.e("floatBuffer[1]", Float.toString(floatBuffer.get(1)));
    Log.e("floatBuffer[2]", Float.toString(floatBuffer.get(2)));
    return Tensor.fromBlob(floatBuffer, new long[] {1, 3, height, width});
  }

  private static String[] readLabels(InputStream is) throws IOException {
    List<String> lines = new ArrayList<String>();
    BufferedReader reader = new BufferedReader(new InputStreamReader(is));
    while (reader.ready()) {
      String line = reader.readLine();
      lines.add(line);
    }
    Log.e("Pytorch", Long.toString(lines.size()));
    return lines.toArray(new String[lines.size()]);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    Bitmap bitmap = null;
    Module module = null;
    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg
      bitmap = BitmapFactory.decodeStream(getAssets().open("image.jpg"));
      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/main/assets/model.pt
      module = Module.load(assetFilePath(this, "model.pt"));
      if (module != null) {
          Log.e("module", "correct");
      }
      InputStream is = getAssets().open("words.txt");
      ImageNetClasses.IMAGENET_CLASSES = readLabels(is);
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }

    // showing image on UI
    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap);

    // preparing input tensor
    final Tensor inputTensor = bitmapToFloat32Tensor(bitmap);
    // running the model
    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

    // getting tensor content as java array of floats
    final float[] scores = outputTensor.getDataAsFloatArray();

    // searching for the index with maximum score
    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = -1;
    Log.e("scores[0]", Float.toString(scores[0]));
    Log.e("scores[1]", Float.toString(scores[1]));
    Log.e("scores[2]", Float.toString(scores[2]));
    Log.e("scores[1000-1]", Float.toString(scores[1000-1]));
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxScoreIdx = i;
      }
    }
    Log.e("maxScoreIdx", Integer.toString(maxScoreIdx));

    String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

    // showing className on UI
    TextView textView = findViewById(R.id.text);
    textView.setText(className);
  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }
}
