package Ep2.Tests;

import Ep2.FrameworkML.ActivationFunctions.Activation;
import Ep2.FrameworkML.ActivationFunctions.Relu;
import Ep2.FrameworkML.ActivationFunctions.Sigmoud;
import Ep2.FrameworkML.CostFunctions.MeanSquareError;
import Ep2.FrameworkML.Layers.Dense;
import Ep2.FrameworkML.Layers.Layer;
import Ep2.FrameworkML.Matrix;
import Ep2.FrameworkML.NeuralNet2;
import Ep2.FrameworkML.Operations;
import org.junit.jupiter.api.Test;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

public class DenseLayerTest {
    @Test
    public void denseLayerInferenceTest(){
        int[] inputShape = new int[]{1, 2, 1};
        int[] outputShape = new int[]{1, 3, 1};
        Layer[] architecture = new Layer[]{
                new Dense(3, new Relu()),
                new Dense(5, new Relu()),
                new Dense(3, new Relu())
        };
        NeuralNet2 nn = new NeuralNet2(inputShape, outputShape, architecture,
                new MeanSquareError());
        Matrix[][] weighAndBias = new Matrix[6][];
        weighAndBias[0] = new Matrix[]{
                new Matrix(2, 3, new float[]{1, 2, 3, 4, 5, 6})
        };
        weighAndBias[1] = new Matrix[]{
                new Matrix(1, 3, new float[]{2, 3, 4})
        };
        weighAndBias[2] = new Matrix[]{
                new Matrix(3, 5, new float[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5})
        };
        weighAndBias[3] = new Matrix[]{
                new Matrix(1, 5, new float[]{5, 4, 3, 2, 1})
        };
        weighAndBias[4] = new Matrix[]{
                new Matrix(5, 3, new float[]{0, 1, 0, 2, 0, 2, 1, 1, 1, 3, 2, 3, 2, 0, 2})
        };
        weighAndBias[5] = new Matrix[]{
                new Matrix(1, 3, new float[]{0, 6, 7})
        };

        for (int i = 0; i < architecture.length; i++){
            architecture[i].setWeights(weighAndBias[i*2]);
            architecture[i].setBiases(weighAndBias[i*2+1]);
        }

        Matrix[] input = new Matrix[]{
                new Matrix(1, 2, new float[]{0, 1})
        };
        Operations.printMatrix(nn.inference(input)[0]);
        assertThat(nn.inference(input)[0].getMatrixArray()).isEqualTo(
                new float[]{715, 306, 722}
        );
        // New Test
        inputShape = new int[]{1, 2, 1};
        outputShape = new int[]{1, 3, 1};
        architecture = new Layer[]{
                new Dense(3, new Relu()),
                new Dense(5, new Relu()),
                new Dense(3, new Sigmoud())
        };
        nn = new NeuralNet2(inputShape, outputShape, architecture,
                new MeanSquareError());
        weighAndBias = new Matrix[6][];
        weighAndBias[0] = new Matrix[]{
                new Matrix(2, 3, new float[]{1, 2, 3, 4, 5, 6})
        };
        weighAndBias[1] = new Matrix[]{
                new Matrix(1, 3, new float[]{2, 3, 4})
        };
        weighAndBias[2] = new Matrix[]{
                new Matrix(3, 5, new float[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5})
        };
        weighAndBias[3] = new Matrix[]{
                new Matrix(1, 5, new float[]{5, 4, 3, 2, 1})
        };
        weighAndBias[4] = new Matrix[]{
                new Matrix(5, 3, new float[]{0, 1, 0, 2, 0, 2, 1, 1, 1, 3, 2, 3, 2, 0, 2})
        };
        weighAndBias[5] = new Matrix[]{
                new Matrix(1, 3, new float[]{0, 6, 7})
        };

        for (int i = 0; i < architecture.length; i++){
            architecture[i].setWeights(weighAndBias[i*2]);
            architecture[i].setBiases(weighAndBias[i*2+1]);
        }

        input = new Matrix[]{
                new Matrix(1, 2, new float[]{0, 1})
        };
        Operations.printMatrix(nn.inference(input)[0]);
        assertThat(nn.inference(input)[0].getMatrixArray()).isEqualTo(
                new float[]{1f, 1f, 1f}
        );
    }

    @Test
    public void denseLayerTrainingTest(){
        int[] inputShape = new int[]{1, 2, 1};
        int[] outputShape = new int[]{1, 3, 1};
        Layer[] architecture = new Layer[]{
                new Dense(3, new Relu()),
                new Dense(5, new Relu()),
                new Dense(3, new Relu())
        };
        NeuralNet2 nn = new NeuralNet2(inputShape, outputShape, architecture,
                new MeanSquareError());
        Matrix[][] weighAndBias = new Matrix[6][];
        weighAndBias[0] = new Matrix[]{
                new Matrix(2, 3, new float[]{1, 2, 3, 4, 5, 6})
        };
        weighAndBias[1] = new Matrix[]{
                new Matrix(1, 3, new float[]{2, 3, 4})
        };
        weighAndBias[2] = new Matrix[]{
                new Matrix(3, 5, new float[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5})
        };
        weighAndBias[3] = new Matrix[]{
                new Matrix(1, 5, new float[]{5, 4, 3, 2, 1})
        };
        weighAndBias[4] = new Matrix[]{
                new Matrix(5, 3, new float[]{0, 1, 0, 2, 0, 2, 1, 1, 1, 3, 2, 3, 2, 0, 2})
        };
        weighAndBias[5] = new Matrix[]{
                new Matrix(1, 3, new float[]{0, 6, 7})
        };

        for (int i = 0; i < architecture.length; i++){
            architecture[i].setWeights(weighAndBias[i*2]);
            architecture[i].setBiases(weighAndBias[i*2+1]);
        }

        Matrix[] input = new Matrix[]{
                new Matrix(1, 2, new float[]{0, 1})
        };
        Matrix[] expected = new Matrix[]{
                new Matrix(1, 3, new float[]{400, 200, 900})
        };

        float cost = nn.cost(new Matrix[][]{input}, new Matrix[][]{expected});
        System.out.println("Initial cost: " + cost);
        assertThat(cost).isEqualTo(71072.5f);
        for (String e : nn.getNetworkString()){
            System.out.println(e);
        }

        System.out.println("TRAINING");
        int trainingCount = 15;
        for (int i = 0; i < trainingCount; i++){
            nn.trainNetwork(new Matrix[][]{input}, new Matrix[][]{expected}, 0.00001f);
            cost = nn.cost(new Matrix[][]{input}, new Matrix[][]{expected});
            System.out.println("Cost After Epoch " + (i+1) + ": " + cost);
        }

        Matrix[] a = nn.inference(input);
        System.out.println("Inference After Training: " + Operations.stringMatrix(a[0]));

        cost = nn.cost(new Matrix[][]{input}, new Matrix[][]{expected});
        System.out.println("Cost After: " + cost);

        for (String e : nn.getNetworkString()){
            System.out.println(e);
        }
    }

    public static void main(String[] args) {





    }


}
