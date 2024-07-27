package Ep2.Tests;

import Ep2.FrameworkML.ActivationFunctions.Relu;
import Ep2.FrameworkML.CostFunctions.MeanSquareError;
import Ep2.FrameworkML.Layers.Conv3D;
import Ep2.FrameworkML.Layers.Layer;
import Ep2.FrameworkML.Matrix;
import Ep2.FrameworkML.NeuralNet2;
import Ep2.FrameworkML.Operations;
import net.sf.saxon.expr.Component;
import org.junit.jupiter.api.Test;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

public class Conv3DLayerTest {
    @Test
    public void inferenceTest(){
        int[] inputShape = new int[]{9, 9, 3};
        int[] outputShape = new int[]{1, 8, 1};
        Layer[] architecture = new Layer[]{
                new Conv3D(3, 3, 2, 3, 3, 1, new Relu()),
                new Conv3D(2, 2,1, 1, 1, 1, new Relu()),
                new Conv3D(1, 1, 1, 1, 1, 1, new Relu(), true),
        };
        // Creating the neural network
        NeuralNet2 nn = new NeuralNet2(inputShape, outputShape, architecture,
                new MeanSquareError());

        // Testing the structure of layer 1.
        Matrix[] oneWeights = architecture[0].getWeights();
        Matrix[] oneBiases = architecture[0].getBiases();
        assertThat(architecture[0].getOutputShape()).isEqualTo(new int[]{3, 3, 2});
        assertThat(oneWeights.length).isEqualTo(2);
        assertThat(oneWeights[0].getRows()).isEqualTo(3);
        assertThat(oneWeights[0].getCols()).isEqualTo(54);
        assertThat(oneBiases.length).isEqualTo(2);
        assertThat(oneBiases[0].getRows()).isEqualTo(3);
        assertThat(oneBiases[0].getCols()).isEqualTo(3);
        Matrix[] w1 = new Matrix[]{
                new Matrix(3, 54, new float[]{
                        0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1,
                        0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1,
                        0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1
                }),
                new Matrix(3, 54, new float[]{
                        0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,
                        0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,
                        0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,
                })
        };
        Matrix[] b1 = new Matrix[]{
                new Matrix(3, 3, new float[]{
                        17, 16, 15,
                        14, 13, 12,
                        11, 10,  9
                }),
                new Matrix(3, 3, new float[]{
                        8, 7, 6,
                        5, 4, 3,
                        2, 1, 0
                })
        };
        architecture[0].setWeights(w1);
        architecture[0].setBiases(b1);

        // Testing structure of layer 2
        Matrix[] twoWeights = architecture[1].getWeights();
        Matrix[] twoBiases = architecture[1].getBiases();

        assertThat(architecture[1].getOutputShape()).isEqualTo(new int[]{2, 2, 2});
        assertThat(twoWeights.length).isEqualTo(1);
        assertThat(twoWeights[0].getRows()).isEqualTo(2);
        assertThat(twoWeights[0].getCols()).isEqualTo(16);
        assertThat(twoBiases.length).isEqualTo(2);
        assertThat(twoBiases[0].getRows()).isEqualTo(2);
        assertThat(twoBiases[0].getCols()).isEqualTo(2);

        Matrix[] w2 = new Matrix[]{
                new Matrix(2, 16, new float[]{
                        -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1,
                        -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1
                }),
        };
        Matrix[] b2 = new Matrix[]{
                new Matrix(2, 2, new float[]{
                        1, 2,
                        3, 4
                }),
                new Matrix(2, 2, new float[]{
                        5, 6,
                        7, 8
                })
        };
        architecture[1].setWeights(w2);
        architecture[1].setBiases(b2);

        // Testing structure of Layer 3
        Matrix[] threeWeights = architecture[2].getWeights();
        Matrix[] threeBiases = architecture[2].getBiases();

        assertThat(architecture[2].getOutputShape()).isEqualTo(new int[]{1, 8, 1});
        assertThat(threeWeights.length).isEqualTo(1);
        assertThat(threeWeights[0].getRows()).isEqualTo(1);
        assertThat(threeWeights[0].getCols()).isEqualTo(8);
        assertThat(threeBiases.length).isEqualTo(2);
        assertThat(threeBiases[0].getRows()).isEqualTo(2);
        assertThat(threeBiases[0].getCols()).isEqualTo(2);

        Matrix[] w3 = new Matrix[]{
                new Matrix(1, 8, new float[]{
                        1, 2, 1, 2, 1, 2, 1, 2
                }),
        };
        Matrix[] b3 = new Matrix[]{
                new Matrix(2, 2, new float[]{
                        1, 0,
                        1, 0
                }),
                new Matrix(2, 2, new float[]{
                        0, 1,
                        0, 1
                })
        };
        architecture[2].setWeights(w3);
        architecture[2].setBiases(b3);

        // Making inputs
        float[] input1 = new float[81];
        float[] input2 = new float[81];
        float[] input3 = new float[81];
        for (int i = 0; i < input1.length; i++){
            input1[i] = i;
            input2[i] = 80 - i;
            input3[i] = 1f;
        }
        Matrix[] input = new Matrix[]{
                // 0 -> 80
                new Matrix(9, 9, input1),
                // 80 -> 0
                new Matrix(9, 9, input2),
                // All 1's
                new Matrix(9, 9, input3)
        };
        // Making inference.
        Matrix[] inference = nn.inference(input);

        assertThat(inference.length).isEqualTo(1);
        assertThat(inference[0].getRows()).isEqualTo(1);
        assertThat(inference[0].getCols()).isEqualTo(8);

        System.out.println("Inference matrix: ");
        for (Matrix m : inference){
            Operations.printMatrix(m);
            System.out.println();
        }

        assertThat(inference.length).isEqualTo(1);
        assertThat(inference[0].getRows()).isEqualTo(1);
        assertThat(inference[0].getCols()).isEqualTo(8);
        assertThat(inference[0].getMatrixArray()).isEqualTo(new float[]{
                1, 0, 1467, 0, 0, 1921, 7, 1
        });

    }

    @Test
    public void trainingTest(){
        int[] inputShape = new int[]{9, 9, 3};
        int[] outputShape = new int[]{1, 8, 1};
        Layer[] architecture = new Layer[]{
                new Conv3D(3, 3, 2, 3, 3, 1, new Relu()),
                new Conv3D(2, 2,1, 1, 1, 1, new Relu()),
                new Conv3D(1, 1, 1, 1, 1, 1, new Relu(), true),
        };
        // Creating the neural network
        NeuralNet2 nn = new NeuralNet2(inputShape, outputShape, architecture,
                new MeanSquareError());

        // Setting parameters for layer 1.
        Matrix[] w1 = new Matrix[]{
                new Matrix(3, 54, new float[]{
                        0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1,
                        0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1,
                        0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1
                }),
                new Matrix(3, 54, new float[]{
                        0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,
                        0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,
                        0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,  0,  0,  0, 1, 1, 1, 2, 2, 2,
                })
        };
        Matrix[] b1 = new Matrix[]{
                new Matrix(3, 3, new float[]{
                        17, 16, 15,
                        14, 13, 12,
                        11, 10,  9
                }),
                new Matrix(3, 3, new float[]{
                        8, 7, 6,
                        5, 4, 3,
                        2, 1, 0
                })
        };
        architecture[0].setWeights(w1);
        architecture[0].setBiases(b1);

        // Setting parameters for layer 2
        Matrix[] w2 = new Matrix[]{
                new Matrix(2, 16, new float[]{
                        -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1,
                        -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1
                }),
        };
        Matrix[] b2 = new Matrix[]{
                new Matrix(2, 2, new float[]{
                        1, 2,
                        3, 4
                }),
                new Matrix(2, 2, new float[]{
                        5, 6,
                        7, 8
                })
        };
        architecture[1].setWeights(w2);
        architecture[1].setBiases(b2);

        // Setting parameters for Layer 3
        Matrix[] w3 = new Matrix[]{
                new Matrix(1, 8, new float[]{
                        1, 2, 1, 2, 1, 2, 1, 2
                }),
        };
        Matrix[] b3 = new Matrix[]{
                new Matrix(2, 2, new float[]{
                        1, 0,
                        1, 0
                }),
                new Matrix(2, 2, new float[]{
                        0, 1,
                        0, 1
                })
        };
        architecture[2].setWeights(w3);
        architecture[2].setBiases(b3);

        // TRAINING TIME
        float[] input1 = new float[81];
        float[] input2 = new float[81];
        float[] input3 = new float[81];
        for (int i = 0; i < input1.length; i++){
            input1[i] = i;
            input2[i] = 80 - i;
            input3[i] = 1f;
        }
        Matrix[] input = new Matrix[]{
                // 0 -> 80
                new Matrix(9, 9, input1),
                // 80 -> 0
                new Matrix(9, 9, input2),
                // All 1's
                new Matrix(9, 9, input3)
        };

        Matrix[] expected = new Matrix[]{
                new Matrix(1, 8, new float[]{
                        1, 1, 1400, 3, 5, 1920, 7, 0
                })
        };
        Matrix[][] trainingIn = new Matrix[][]{
                input,
        };
        Matrix[][] trainingEx = new Matrix[][]{
                expected,
        };

        System.out.println("This should equal 2263: " + nn.cost(trainingIn, trainingEx));
        nn.trainNetwork(trainingIn, trainingEx, 0.0000000001f);

    }



    public static void main(String[] args) {
        int[] values = new int[]{
                80, 79, 78,
                71, 70, 69,
                62, 61, 60
        };
        int[] values2 = new int[]{
                1, 1, 1,
                1, 1, 1,
                1, 1, 1
        };
        addStride(-60, values);
        int filterVal = 1;
        System.out.println(p(filterVal, values) + p(filterVal+1, values2));
    }
    private static int p(int filterValue, int[] values){
        int cum = 0;
        for (int i : values){
            cum += filterValue*i;
        }
        return cum;
    }
    private static void addStride(int stride, int[] values){
        for (int i = 0; i < values.length; i++){
            values[i] = values[i]+stride;
        }
    }



}
