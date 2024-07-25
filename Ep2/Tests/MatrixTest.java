package Ep2.Tests;

import Ep2.FrameworkML.Matrix;
import org.junit.jupiter.api.Test;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

public class MatrixTest {
    @Test
    public void getElementTest(){
        // getElement Tests
        Matrix a = new Matrix(1, 1, new float[]{1f});
        assertThat(a.getElement(1, 1)).isEqualTo(1f);

        Matrix b = new Matrix(2, 2, new float[]{1f, 2f, 3f, 4f});
        assertThat(b.getElement(2, 1)).isEqualTo(3f);
        assertThat(b.getElement(2, 2)).isEqualTo(4f);

        Matrix c = new Matrix(4, 3, new float[]{1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9,
                10f, 11f, 12f});

        assertThat(c.getElement(1, 3)).isEqualTo(3f);
        assertThat(c.getElement(4, 1)).isEqualTo(10f);
        assertThat(c.getElement(3, 2)).isEqualTo(8f);
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> c.getElement(5, 1));
        assertThat(e).isInstanceOf(IllegalArgumentException.class);

        IllegalArgumentException e2 =
                assertThrows(IllegalArgumentException.class, () -> c.getElement(1, 4));
        assertThat(e2).isInstanceOf(IllegalArgumentException.class);

        IllegalArgumentException e3 =
                assertThrows(IllegalArgumentException.class, () -> c.getElement(0, 2));
        assertThat(e3).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    public void getColAtTest(){
        Matrix a = new Matrix(1, 4, new float[]{
                1f, 2f, 3f, 4f
        });
        assertThat(a.getColAt(1).getMatrixArray()).isEqualTo(new float[]{1f});
        assertThat(a.getColAt(1).getCols()).isEqualTo(1);
        assertThat(a.getColAt(1).getRows()).isEqualTo(1);

        assertThat(a.getColAt(2).getMatrixArray()).isEqualTo(new float[]{2f});
        assertThat(a.getColAt(3).getMatrixArray()).isEqualTo(new float[]{3f});
        assertThat(a.getColAt(4).getMatrixArray()).isEqualTo(new float[]{4f});

        // Accessing row 0 and 5 should result in error
        IllegalArgumentException e1 =
                assertThrows(IllegalArgumentException.class, () -> a.getRowAt(0));
        assertThat(e1).isInstanceOf(IllegalArgumentException.class);
        IllegalArgumentException e2 =
                assertThrows(IllegalArgumentException.class, () -> a.getRowAt(5));
        assertThat(e2).isInstanceOf(IllegalArgumentException.class);

        Matrix b = new Matrix(3, 3, new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f
        });
        assertThat(b.getColAt(1).getMatrixArray()).isEqualTo(new float[]{
                1f, 4f, 7f
        });
        assertThat(b.getColAt(2).getMatrixArray()).isEqualTo(new float[]{
                2f, 5f, 8f
        });
        assertThat(b.getColAt(2).getCols()).isEqualTo(1);
        assertThat(b.getColAt(2).getRows()).isEqualTo(3);
        assertThat(b.getColAt(3).getMatrixArray()).isEqualTo(new float[]{
                3f, 6f, 9f
        });

        IllegalArgumentException e3 =
                assertThrows(IllegalArgumentException.class, () -> b.getColAt(0));
        assertThat(e3).isInstanceOf(IllegalArgumentException.class);
        IllegalArgumentException e4 =
                assertThrows(IllegalArgumentException.class, () -> b.getColAt(4));
        assertThat(e4).isInstanceOf(IllegalArgumentException.class);


    }

    @Test
    public void getRowAtTest(){
        Matrix a = new Matrix(1, 4, new float[]{
                1f, 2f, 3f, 4f
        });
        assertThat(a.getRowAt(1).getMatrixArray()).isEqualTo(new float[]{
                1f, 2f, 3f, 4f
        });
        assertThat(a.getRowAt(1).getCols()).isEqualTo(4);
        assertThat(a.getRowAt(1).getRows()).isEqualTo(1);

        // Accessing row 0 and 2 should result in error
        IllegalArgumentException e1 =
                assertThrows(IllegalArgumentException.class, () -> a.getRowAt(0));
        assertThat(e1).isInstanceOf(IllegalArgumentException.class);
        IllegalArgumentException e2 =
                assertThrows(IllegalArgumentException.class, () -> a.getRowAt(2));
        assertThat(e2).isInstanceOf(IllegalArgumentException.class);

        Matrix b = new Matrix(3, 3, new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f
        });
        assertThat(b.getRowAt(1).getMatrixArray()).isEqualTo(new float[]{
                1f, 2f, 3f
        });
        assertThat(b.getRowAt(2).getMatrixArray()).isEqualTo(new float[]{
                4f, 5f, 6f
        });
        assertThat(b.getRowAt(2).getCols()).isEqualTo(3);
        assertThat(b.getRowAt(2).getRows()).isEqualTo(1);

        assertThat(b.getRowAt(3).getMatrixArray()).isEqualTo(new float[]{
                7f, 8f, 9f
        });

        IllegalArgumentException e3 =
                assertThrows(IllegalArgumentException.class, () -> b.getRowAt(0));
        assertThat(e3).isInstanceOf(IllegalArgumentException.class);
        IllegalArgumentException e4 =
                assertThrows(IllegalArgumentException.class, () -> a.getRowAt(4));
        assertThat(e4).isInstanceOf(IllegalArgumentException.class);


    }
    
}
