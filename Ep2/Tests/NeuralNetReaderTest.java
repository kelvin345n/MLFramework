package Ep2.Tests;

import Ep2.FrameworkML.*;
import org.junit.jupiter.api.Test;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

public class NeuralNetReaderTest {

    @Test
    public void readIntArrayTest(){
        String str = "[1, 2, 3, 4, 5, 6, 7, 100, 24241, 11]";
        assertThat(NeuralNetReader.readIntArray(str)).isEqualTo(new int[]{1, 2, 3, 4, 5, 6, 7, 100, 24241, 11});
    }
    @Test
    public void readFloatArrayTest(){
        String str = "[0.1309581, 0.3134314, 1.2414, 4.1313, 5.444, 6.756244253, 7.9578134, 100.038429753, 24241.13, 11.0]";
        assertThat(NeuralNetReader.readFloatArray(str)).isEqualTo(new float[]{
                0.1309581f, 0.3134314f, 1.2414f, 4.1313f, 5.444f, 6.756244253f, 7.9578134f, 100.038429753f, 24241.13f, 11.0f
        });
    }

    @Test
    public void readMatrixTest(){
        String str = "3-4-[3.14, 3.145, 3.14151, 0.35153, 0.135351, 0.6744, 0.35454, 5.131, 0.111, .122, .8888, .999]";
        Matrix a = NeuralNetReader.readMatrix(str);
        assertThat(a.getRows()).isEqualTo(3);
        assertThat(a.getCols()).isEqualTo(4);
        assertThat(a.getMatrixArray()).isEqualTo(new float[]{
                3.14f, 3.145f, 3.14151f, 0.35153f, 0.135351f, 0.6744f, 0.35454f, 5.131f, 0.111f, .122f, .8888f, .999f
        });
    }
}
