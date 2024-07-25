package Ep2.Tests;

import Ep2.FrameworkML.*;
import org.junit.jupiter.api.Test;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

public class OperationsTest {
    @Test
    public void sumMatrixTest() {
        Matrix a = new Matrix(3, 3, new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});
        Matrix b = new Matrix(3, 3, new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f});

        Operations.sumMatrix(a, b);
        assertThat(a.getMatrixArray()).isEqualTo(new float[]{
                2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f, 18f
        });
        assertThat(b.getMatrixArray()).isEqualTo(new float[]{
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f
        });


        Operations.sumMatrix(b, a);
        assertThat(a.getMatrixArray()).isEqualTo(new float[]{
                2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f, 18f
        });
        assertThat(b.getMatrixArray()).isEqualTo(new float[]{
                3f, 6f, 9f, 12f, 15f, 18f, 21f, 24f, 27f
        });

        Matrix c = new Matrix(4, 1, new float[]{1f, 2f, 3f, 4f});
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> Operations.sumMatrix(a, c));
        assertThat(e).isInstanceOf(IllegalArgumentException.class);
        IllegalArgumentException e2 =
                assertThrows(IllegalArgumentException.class, () -> Operations.sumMatrix(c, b));
        assertThat(e2).isInstanceOf(IllegalArgumentException.class);


        Matrix d = new Matrix(4, 3, new float[]{1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9,
                10f, 11f, 12f});

        Matrix f = new Matrix(4, 3, new float[]{
                12f, 11f, 10f, 9f, 8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f
        });

        Operations.sumMatrix(d, f);
        assertThat(d.getMatrixArray()).isEqualTo(new float[]{
                13f, 13f, 13f, 13f, 13f, 13f, 13f, 13f, 13f, 13f, 13f, 13f,
        });

        IllegalArgumentException e3 =
                assertThrows(IllegalArgumentException.class, () -> Operations.sumMatrix(d, b));
        assertThat(e3).isInstanceOf(IllegalArgumentException.class);

        assertThat(f.getMatrixArray()).isEqualTo(new float[]{
                12f, 11f, 10f, 9f, 8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f
        });


    }

    @Test
    public void dotMatrixTest(){
        Matrix a = new Matrix(2, 4, new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f
        });
        Matrix b = new Matrix(4, 2, new float[]{
                1f, 2f,
                3f, 4f,
                5f, 6f,
                7f, 8f
        });

        Matrix ab = Operations.dotMatrix(a, b);
        assertThat(ab.getMatrixArray()).isEqualTo(new float[]{50, 60, 114, 140});
        assertThat(ab.getRows()).isEqualTo(2);
        assertThat(ab.getCols()).isEqualTo(2);

        Matrix ba = Operations.dotMatrix(b, a);
        assertThat(ba.getMatrixArray()).isEqualTo(new float[]{
                11f, 14f, 17f, 20f,
                23f, 30f, 37f, 44f,
                35f, 46f, 57f, 68f,
                47f, 62f, 77f, 92f
        });
        assertThat(ba.getRows()).isEqualTo(4);
        assertThat(ba.getCols()).isEqualTo(4);


        // More testing
        Matrix c = new Matrix(3, 4, new float[]{
                -1f, 0f, 5f, -3f,
                2f,  2f, 1f, -2f,
                4f, 0f, -2f, -3f
        });
        Matrix d = new Matrix(4, 1, new float[]{
                1f,
                0f,
                2f,
                3f,
        });
        Matrix cd = Operations.dotMatrix(c, d);
        assertThat(cd.getMatrixArray()).isEqualTo(new float[]{0, -2, -9});
        assertThat(cd.getRows()).isEqualTo(3);
        assertThat(cd.getCols()).isEqualTo(1);

        //d dot c should be an error
        IllegalArgumentException e2 =
                assertThrows(IllegalArgumentException.class, () -> Operations.dotMatrix(d, c));
        assertThat(e2).isInstanceOf(IllegalArgumentException.class);
        // a dot c should error
        IllegalArgumentException e3 =
                assertThrows(IllegalArgumentException.class, () -> Operations.dotMatrix(a, c));
        assertThat(e3).isInstanceOf(IllegalArgumentException.class);

        // c dot b
        Matrix cb = Operations.dotMatrix(c, b);
        assertThat(cb.getMatrixArray()).isEqualTo(new float[]{
                3f, 4f,
                -1f, 2f,
                -27f, -28f
        });
        assertThat(cb.getRows()).isEqualTo(3);
        assertThat(cb.getCols()).isEqualTo(2);

        // Big testing
        Matrix e = new Matrix(7, 3, new float[]{
                1f, -1f, 0f,
                0f, 1f, -3f,
                -2f, 2f, 0f,
                0f, 0f, 1f,
                -1f, -3f, 2f,
                2f, 1f, 1f,
                3f, 3f, -1f

        });
        Matrix f = new Matrix(3, 3, new float[]{
                0f, 1f, 3f,
                -1f, 2f, 3f,
                1f, -2f, -3f
        });

        // c dot b
        Matrix ef = Operations.dotMatrix(e, f);
        assertThat(ef.getMatrixArray()).isEqualTo(new float[]{
                1f, -1f, 0f,
                -4f, 8f, 12f,
                -2f, 2f, 0f,
                1f, -2f, -3f,
                5f, -11f, -18f,
                0f, 2f, 6f,
                -4f, 11f, 21f
        });
        assertThat(ef.getRows()).isEqualTo(7);
        assertThat(ef.getCols()).isEqualTo(3);

        // f dot e should error
        IllegalArgumentException e4 =
                assertThrows(IllegalArgumentException.class, () -> Operations.dotMatrix(f, e));
        assertThat(e4).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    public void hagamardTest(){
        Matrix a = new Matrix(2, 4, new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f
        });
        Matrix b = new Matrix(2, 4, new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f
        });

        Matrix ab = Operations.hadamard(a, b);
        assertThat(ab.getMatrixArray()).isEqualTo(new float[]{1f, 4f, 9f, 16f,
                                                              25f, 36f, 49f, 64f});
        assertThat(ab.getRows()).isEqualTo(2);
        assertThat(ab.getCols()).isEqualTo(4);

        Matrix c = new Matrix(4, 2, new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f
        });
        Matrix d = new Matrix(4, 2, new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f
        });
        Matrix cd = Operations.hadamard(c, d);
        assertThat(cd.getMatrixArray()).isEqualTo(new float[]{1f, 4f, 9f, 16f,
                                                        25f, 36f, 49f, 64f});
        assertThat(cd.getRows()).isEqualTo(4);
        assertThat(cd.getCols()).isEqualTo(2);

        Matrix e = new Matrix(3, 3, new float[]{
                0f, 1f, 0f,
                1f, 0f, 1f,
                0f, 1f, 0f
        });
        Matrix fail = new Matrix(4, 2, new float[]{
                2f, 2f, 2f, 2f,
                2f, 2f, 2f, 2f
        });
        IllegalArgumentException err =
                assertThrows(IllegalArgumentException.class, () -> Operations.hadamard(e, fail));
        assertThat(err).isInstanceOf(IllegalArgumentException.class);

        Matrix f = new Matrix(3, 3, new float[]{
                2f, 2f, 2f,
                2f, 2f, 2f,
                2f, 2f, 2f
        });
        Matrix ef = Operations.hadamard(e, f);
        Matrix fe = Operations.hadamard(f, e);
        assertThat(ef.getMatrixArray()).isEqualTo(fe.getMatrixArray());
        assertThat(ef.getMatrixArray()).isEqualTo(new float[]{
                0f, 2f, 0f,
                2f, 0f, 2f,
                0f, 2f, 0f});
        assertThat(ef.getRows()).isEqualTo(3);
        assertThat(ef.getCols()).isEqualTo(3);
    }

    @Test
    public void createSubMatrixTest() {
        Matrix a = new Matrix(8, 8, new float[]{
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f,
                17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f,
                25f, 26f, 27f, 28f, 29f, 30f, 31f, 32f,
                33f, 34f, 35f, 36f, 37f, 38f, 39f, 40f,
                41f, 42f, 43f, 44f, 45f, 46f, 47f, 48f,
                49f, 50f, 51f, 52f, 53f, 54f, 55f, 56f,
                57f, 58f, 59f, 60f, 61f, 62f, 63f, 64f
        });

        Matrix aSub1 = Operations.createSubMatrix(a, 1, 1, 3, 3);
        assertThat(aSub1.getMatrixArray()).isEqualTo(new float[]{1f, 2f, 3f,
                9f, 10f, 11f,
                17f, 18f, 19f});

        Matrix aSub2 = Operations.createSubMatrix(a, 3, 4, 5, 5);
        assertThat(aSub2.getMatrixArray()).isEqualTo(new float[]{20f, 21f, 22f, 23f, 24f,
                28f, 29f, 30f, 31f, 32f,
                36f, 37f, 38f, 39f, 40f,
                44f, 45f, 46f, 47f, 48f,
                52f, 53f, 54f, 55f, 56f});

        Matrix aSub3 = Operations.createSubMatrix(a, 3, 4, 6, 4);
        assertThat(aSub3.getMatrixArray()).isEqualTo(new float[]{20f, 21f, 22f, 23f,
                28f, 29f, 30f, 31f,
                36f, 37f, 38f, 39f,
                44f, 45f, 46f, 47f,
                52f, 53f, 54f, 55f,
                60f, 61f, 62f, 63f,});

        Matrix aSub4 = Operations.createSubMatrix(a, 1, 1, 8, 8);
        assertThat(aSub4.getMatrixArray()).isEqualTo(a.getMatrixArray());

        Matrix aSub5 = Operations.createSubMatrix(a, 8, 1, 1, 1);
        assertThat(aSub5.getMatrixArray()).isEqualTo(new float[]{57f});

        Matrix aSub6 = Operations.createSubMatrix(a, 8, 8, 1, 1);
        assertThat(aSub6.getMatrixArray()).isEqualTo(new float[]{64f});

        IllegalArgumentException err =
                assertThrows(IllegalArgumentException.class, () -> Operations.createSubMatrix(a, 0, 1, 6, 4));
        assertThat(err).isInstanceOf(IllegalArgumentException.class);

        IllegalArgumentException err2 =
                assertThrows(IllegalArgumentException.class, () -> Operations.createSubMatrix(a, 1, 0, 3, 1));
        assertThat(err2).isInstanceOf(IllegalArgumentException.class);

        IllegalArgumentException err3 =
                assertThrows(IllegalArgumentException.class, () -> Operations.createSubMatrix(a, 9, 1, 1, 1));
        assertThat(err3).isInstanceOf(IllegalArgumentException.class);

        IllegalArgumentException err4 =
                assertThrows(IllegalArgumentException.class, () -> Operations.createSubMatrix(a, 3, 9, 2, 2));
        assertThat(err4).isInstanceOf(IllegalArgumentException.class);

        IllegalArgumentException err5 =
                assertThrows(IllegalArgumentException.class, () -> Operations.createSubMatrix(a, 1, 7, 3, 3));
        assertThat(err5).isInstanceOf(IllegalArgumentException.class);

        IllegalArgumentException err6 =
                assertThrows(IllegalArgumentException.class, () -> Operations.createSubMatrix(a, 5, 7, 5, 1));
        assertThat(err6).isInstanceOf(IllegalArgumentException.class);

        Matrix b = new Matrix(3, 6, new float[]{
                1f, 2f, 3f, 4f, 5f, 6f,
                7f, 8f, 9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f, 17f, 18f});

        Matrix bSub1 = Operations.createSubMatrix(b, 1, 5, 2, 2);
        assertThat(bSub1.getMatrixArray()).isEqualTo(new float[]{5f, 6f, 11f, 12f});
        IllegalArgumentException err7 =
                assertThrows(IllegalArgumentException.class, () -> Operations.createSubMatrix(b, 1, 6, 2, 2));
        assertThat(err7).isInstanceOf(IllegalArgumentException.class);

        Matrix bSub2 = Operations.createSubMatrix(b, 3, 1, 1, 6);
        assertThat(bSub2.getMatrixArray()).isEqualTo(new float[]{13f, 14f, 15f, 16f, 17f, 18f});
        IllegalArgumentException err8 =
                assertThrows(IllegalArgumentException.class, () -> Operations.createSubMatrix(b, 3, 1, 2, 6));
        assertThat(err8).isInstanceOf(IllegalArgumentException.class);

        Matrix bSub3 = Operations.createSubMatrix(b, 2, 2, 2, 2);
        assertThat(bSub3.getMatrixArray()).isEqualTo(new float[]{8f, 9f, 14f, 15f});
        IllegalArgumentException err9 =
                assertThrows(IllegalArgumentException.class, () -> Operations.createSubMatrix(b, 4, 2, 2, 2));
        assertThat(err9).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    public void elementSumTest(){
        Matrix a = new Matrix(2, 8, new float[]{
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f,
        });
        assertThat(Operations.elementSum(a)).isEqualTo(136f);

        a = new Matrix(6, 1, new float[]{
                1f, 2f, 3f, 4f, 5f, 6f
        });
        assertThat(Operations.elementSum(a)).isEqualTo(21f);

        a = new Matrix(8, 8, new float[]{
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f,
                17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f,
                25f, 26f, 27f, 28f, 29f, 30f, 31f, 32f,
                33f, 34f, 35f, 36f, 37f, 38f, 39f, 40f,
                41f, 42f, 43f, 44f, 45f, 46f, 47f, 48f,
                49f, 50f, 51f, 52f, 53f, 54f, 55f, 56f,
                57f, 58f, 59f, 60f, 61f, 62f, 63f, 64f
        });
        assertThat(Operations.elementSum(a)).isEqualTo(2080f);

        a = new Matrix(3, 14, new float[]{
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f,
                15f, 16f, 17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f, 25f, 26f, 27f, 28f,
                29f, 30f, 31f, 32f, 33f, 34f, 35f, 36f, 37f, 38f, 39f, 40f, 41f, 42f,
        });
        assertThat(Operations.elementSum(a)).isEqualTo(903f);
    }

    @Test
    public void flattenTest(){
        Matrix[] a = new Matrix[]{
                new Matrix(2, 4, new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}),
                new Matrix(2, 4, new float[]{9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f}),
                new Matrix(2, 4, new float[]{17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f}),
        };
        Matrix[] aOut = Operations.flatten(a);
        assertThat(aOut.length).isEqualTo(1);
        assertThat(aOut[0].getRows()).isEqualTo(1);
        assertThat(aOut[0].getCols()).isEqualTo(24);
        assertThat(aOut[0].getMatrixArray()).isEqualTo(new float[]{
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f
        });

        Matrix[] b = new Matrix[]{
                new Matrix(1, 1, new float[]{1f}),
                new Matrix(1, 1, new float[]{2f}),
                new Matrix(1, 1, new float[]{3f}),
                new Matrix(1, 1, new float[]{4f}),
                new Matrix(1, 1, new float[]{5f}),
        };
        Matrix[] bOut = Operations.flatten(a);
        assertThat(bOut.length).isEqualTo(1);
        assertThat(bOut[0].getRows()).isEqualTo(1);
        assertThat(bOut[0].getCols()).isEqualTo(5);
        assertThat(bOut[0].getMatrixArray()).isEqualTo(new float[]{
                1f, 2f, 3f, 4f, 5f
        });

        Matrix[] c = new Matrix[]{
                new Matrix(2, 1, new float[]{1f, 2f}),
                new Matrix(2, 1, new float[]{3f, 4f}),
                new Matrix(2, 1, new float[]{5f, 6f}),
                new Matrix(2, 1, new float[]{7f, 8f}),
                new Matrix(2, 1, new float[]{9f, 10f}),
        };
        Matrix[] cOut = Operations.flatten(a);
        assertThat(cOut.length).isEqualTo(1);
        assertThat(cOut[0].getRows()).isEqualTo(1);
        assertThat(cOut[0].getCols()).isEqualTo(10);
        assertThat(cOut[0].getMatrixArray()).isEqualTo(new float[]{
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f
        });
    }

    @Test
    public void reshapeTest(){
        Matrix[] a = new Matrix[]{new Matrix(1, 24, new float[]{
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f
        })};
        Matrix[] aOut = Operations.reshape(a, new int[]{2, 4, 3});

        assertThat(aOut.length).isEqualTo(3);
        assertThat(aOut[0].getMatrixArray()).isEqualTo(new float[]{
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f
        });
        assertThat(aOut[1].getMatrixArray()).isEqualTo(new float[]{
                9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f
        });
        assertThat(aOut[2].getMatrixArray()).isEqualTo(new float[]{
                17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f
        });
        for (int d = 0; d < 3; d++){
           assertThat(aOut[d].getRows()).isEqualTo(2);
           assertThat(aOut[d].getCols()).isEqualTo(4);
        }

        Matrix[] b = new Matrix[]{
                new Matrix(1, 1, new float[]{1f}),
                new Matrix(1, 1, new float[]{2f}),
                new Matrix(1, 1, new float[]{3f}),
                new Matrix(1, 1, new float[]{4f}),
                new Matrix(1, 1, new float[]{5f}),
        };
        Matrix[] bOut = Operations.reshape(b, new int[]{1, 5, 1});
        assertThat(bOut.length).isEqualTo(1);
        assertThat(bOut[0].getRows()).isEqualTo(1);
        assertThat(bOut[0].getCols()).isEqualTo(5);
        assertThat(bOut[0].getMatrixArray()).isEqualTo(new float[]{
                1f, 2f, 3f, 4f, 5f
        });

        // Test 3
        Matrix[] c = new Matrix[5];
        for (int i = 0; i < 5; i++){
            float[] array = new float[70];
            for (int j = 0; j < 70; j++){
                array[j] = (i * 70) + j;
            }
            c[i] = new Matrix(10, 7, array);
        }

        IllegalArgumentException err =
                assertThrows(IllegalArgumentException.class, () -> Operations.reshape(c, new int[]{1, 4, 5}));
        assertThat(err).isInstanceOf(IllegalArgumentException.class);

        Matrix[] cOut = Operations.reshape(c, new int[]{5, 10, 7});


        assertThat(cOut.length).isEqualTo(7);
        for (int d = 0; d < 7; d++){
            assertThat(cOut[d].getRows()).isEqualTo(5);
            assertThat(cOut[d].getCols()).isEqualTo(10);
        }
        for (int i = 0; i < 7; i++){
            float[] array = new float[50];
            for (int j = 0; j < 50; j++){
                array[j] = (i * 50) + j;
            }
            assertThat(cOut[i].getMatrixArray()).isEqualTo(array);
        }

        for (Matrix m : cOut){
            Operations.printMatrix(m);
            System.out.println();
        }






    }

    @Test
    public void createSubTensorTest(){
        // Creating a 4x4x4 tensor.
        Matrix[] a = new Matrix[4];
        // Where first layer will be numbered 1-16, second 17-32...etc
        int rowSize = 4;
        int colSize = 4;
        // Populating each matrix
        for (int d = 0; d < a.length; d++){
            float[] input = new float[rowSize*colSize];
            for (int i = 1; i <= rowSize*colSize; i++){
                input[i-1] = i + (d*rowSize*colSize);
            }
            a[d] = new Matrix(rowSize, colSize, input);
        }

        for (Matrix m : a){
            Operations.printMatrix(m);
            System.out.println();
        }

        Matrix[] aSub = Operations.createSubTensor(a, 1, 1, 0, 2, 2, 2);
        assertThat(aSub.length).isEqualTo(2);
        for (int i = 0; i < aSub.length; i++){
            assertThat(aSub[i].getRows()).isEqualTo(2);
            assertThat(aSub[i].getCols()).isEqualTo(2);
            assertThat(aSub[i].getMatrixArray()).isEqualTo(new float[]{
                    1f+(i*colSize*rowSize), 2f+(i*colSize*rowSize),
                    5f+(i*colSize*rowSize), 6f+(i*colSize*rowSize)
            });
        }

        Matrix[] aSub2 = Operations.createSubTensor(a, 2, 4, 1, 3, 1, 3);
        assertThat(aSub2.length).isEqualTo(3);
        for (int i = 0; i < aSub2.length; i++){
            assertThat(aSub2[i].getRows()).isEqualTo(3);
            assertThat(aSub2[i].getCols()).isEqualTo(1);
            assertThat(aSub2[i].getMatrixArray()).isEqualTo(new float[]{
                    24f+(i*16), 28f+(i*16),
                    32f+(i*16)
            });
        }


        Matrix[] aSub3 = Operations.createSubTensor(a, 1, 1, 3, 4, 4, 1);
        assertThat(aSub3.length).isEqualTo(1);
        for (int i = 0; i < aSub3.length; i++){
            assertThat(aSub3[i].getRows()).isEqualTo(4);
            assertThat(aSub3[i].getCols()).isEqualTo(4);
            assertThat(aSub3[i].getMatrixArray()).isEqualTo(a[3].getMatrixArray());
        }

        // Error testing.
        IllegalArgumentException err =
                assertThrows(IllegalArgumentException.class, () ->
                        Operations.createSubTensor(a, 3, 1, 0, 3, 2, 2));
        assertThat(err).isInstanceOf(IllegalArgumentException.class);

        IllegalArgumentException err2 =
                assertThrows(IllegalArgumentException.class, () ->
                        Operations.createSubTensor(a, 1, 2, 0, 2, 4, 2));
        assertThat(err2).isInstanceOf(IllegalArgumentException.class);

        IllegalArgumentException err3 =
                assertThrows(IllegalArgumentException.class, () ->
                        Operations.createSubTensor(a, 1, 1, 3, 2, 2, 2));
        assertThat(err3).isInstanceOf(IllegalArgumentException.class);


    }

    @Test
    public void addSubTensorTest(){
        // Creating a 4x4x4 tensor.
        Matrix[] a = new Matrix[4];
        // Where first layer will be numbered 1-16, second 17-32...etc
        int rowSize = 4;
        int colSize = 4;
        // Populating each matrix
        for (int d = 0; d < a.length; d++){
            float[] input = new float[rowSize*colSize];
            for (int i = 1; i <= rowSize*colSize; i++){
                input[i-1] = i + (d*rowSize*colSize);
            }
            a[d] = new Matrix(rowSize, colSize, input);
        }

        // Creating a 4x4x4 tensor.
        Matrix[] aSub = new Matrix[4];
        // Where first layer will be numbered 1-16, second 17-32...etc
        rowSize = 4;
        colSize = 4;
        // Populating each matrix
        for (int d = 0; d < aSub.length; d++){
            float[] input = new float[rowSize*colSize];
            for (int i = 1; i <= rowSize*colSize; i++){
                input[i-1] = i + (d*rowSize*colSize);
            }
            aSub[d] = new Matrix(rowSize, colSize, input);
        }

        // Error testing.
        IllegalArgumentException err =
                assertThrows(IllegalArgumentException.class, () ->
                        Operations.addSubTensor(a, aSub, 1, 1, 1));
        assertThat(err).isInstanceOf(IllegalArgumentException.class);

        err = assertThrows(IllegalArgumentException.class, () ->
                        Operations.addSubTensor(a, aSub, 2, 1, 0));
        assertThat(err).isInstanceOf(IllegalArgumentException.class);

        err = assertThrows(IllegalArgumentException.class, () ->
                        Operations.addSubTensor(a, aSub, 1, 2, 0));
        assertThat(err).isInstanceOf(IllegalArgumentException.class);

        Operations.addSubTensor(a, aSub, 1, 1, 0);

        // a is now changed.
        for (int d = 0; d < a.length; d++){
            float[] test = new float[rowSize*colSize];
            for (int i = 1; i <= rowSize*colSize; i++){
                test[i-1] = (i + (d*rowSize*colSize))*2;
            }
            assertThat(a[d].getMatrixArray()).isEqualTo(test);
        }

        // New Test

        Matrix[] aSub2 = new Matrix[2];
        rowSize = 2;
        colSize = 2;
        // Populating each matrix
        for (int d = 0; d < aSub2.length; d++){
            float[] input = new float[rowSize*colSize];
            for (int i = 1; i <= rowSize*colSize; i++){
                input[i-1] = 1f;
            }
            aSub2[d] = new Matrix(rowSize, colSize, input);
        }

        Operations.addSubTensor(a, aSub2, 3, 3, 2);

        // a is now changed.

        for (Matrix m : a){
            Operations.printMatrix(m);
            System.out.println();
        }





    }



}
