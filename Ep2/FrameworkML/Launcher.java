package Ep2.FrameworkML;

import Ep2.FrameworkML.ActivationFunctions.Activation;
import Ep2.FrameworkML.ActivationFunctions.Relu;
import Ep2.FrameworkML.ActivationFunctions.Sigmoud;
import Ep2.FrameworkML.ActivationFunctions.Softmax;
import Ep2.FrameworkML.CostFunctions.BinaryCrossEntropy;
import Ep2.FrameworkML.CostFunctions.MeanSquareError;
import Ep2.FrameworkML.Layers.Dense;
import Ep2.FrameworkML.Layers.Layer;
import net.sf.saxon.expr.Component;

import java.util.Arrays;

public class Launcher {



    public static void main(String[] strings){

//        Matrix m = new Matrix(10, 10);
//        m.randomizeMatrix(0, 1);
//        Operations.printMatrix(m);
//        int count = 0;
//        int[] hidLayer = new int[]{2, 1};
//        Matrix trainData = new Matrix(4, 2, new float[]{
//                0, 0,
//                0, 1,
//                1, 0,
//                1, 1
//        });
//        Matrix expected = new Matrix(4, 1, new float[]{0, 1, 1, 0});
//
//        for (int i = 0; i < 100; i++){
//            NeuralNet nn = new NeuralNet(2, 1, hidLayer, trainData,
//                    expected, 0, 1, new Sigmoud());
//            Matrix testMat = trainData;
//            nn.trainNeuralNet(25000, 1e-1f, 1e-1f);
//            if (nn.cost(new Sigmoud()) > 0.05f){
//                count++;
//            }
//        }
//        System.out.println("Count: " + count);
/**  FOIENIJNIWJF
 *
 *
 *
 *
 * EKEJW */
        int[] inputShape = new int[]{1, 2, 1}; // 1 row, 2 cols, 1 matrix
        int[] outputShape = new int[]{1, 1, 1};
        Layer[] architecture = new Layer[]{
                new Dense(2, new Sigmoud()),
                new Dense(1, new Sigmoud())
        };
        NeuralNet2 xor = new NeuralNet2(inputShape, outputShape, architecture,
                new BinaryCrossEntropy());
        Matrix[] w1 = new Matrix[]{
                new Matrix(2, 2, new float[]{0.4f, 0.5f, 0.6f, 0.7f})
        };
        Matrix[] b1 = new Matrix[]{
                new Matrix(1, 2, new float[]{0.1f, 0.9f})
        };
        Matrix[] w2 = new Matrix[]{
            new Matrix(2, 1, new float[]{0.5f, 0.4f})
        };
        Matrix[] b2 = new Matrix[]{
                new Matrix(1, 1, new float[]{0.3f})
        };

        architecture[0].setWeights(w1);
        architecture[0].setBiases(b1);
        architecture[1].setWeights(w2);
        architecture[1].setBiases(b2);

        for (String s : xor.getNetworkString()){
            System.out.println(s);
        }

        Matrix[][] trainInputs = new Matrix[4][];
        trainInputs[0] = new Matrix[]{new Matrix(1, 2, new float[]{0f, 0f})};
        trainInputs[1] = new Matrix[]{new Matrix(1, 2, new float[]{0f, 1f})};
        trainInputs[2] = new Matrix[]{new Matrix(1, 2, new float[]{1f, 0f})};
        trainInputs[3] = new Matrix[]{new Matrix(1, 2, new float[]{1f, 1f})};

        Matrix[][] trainOut = new Matrix[4][];
        trainOut[0] = new Matrix[]{new Matrix(1, 1, new float[]{0f})};
        trainOut[1] = new Matrix[]{new Matrix(1, 1, new float[]{1f})};
        trainOut[2] = new Matrix[]{new Matrix(1, 1, new float[]{1f})};
        trainOut[3] = new Matrix[]{new Matrix(1, 1, new float[]{0f})};

        System.out.println("Initial cost: " + xor.cost(trainInputs, trainOut));
        for (int i = 0; i < 10000; i++) {
            float lr = 0.1f;
            if (i > 4000){
                lr = 1f;
            }
            xor.trainNetwork(trainInputs, trainOut, lr);
            System.out.println("Cost after training: " + xor.cost(trainInputs, trainOut));
        }
        System.out.println("Cost: " + xor.cost(trainInputs, trainOut));
        System.out.println("----INFERENCES----");
        System.out.println("(0, 0): ");
        Operations.printMatrix(xor.inference(trainInputs[0])[0]);
        System.out.println("(1, 0): ");
        Operations.printMatrix(xor.inference(trainInputs[1])[0]);
        System.out.println("(0, 1): ");
        Operations.printMatrix(xor.inference(trainInputs[2])[0]);
        System.out.println("(1, 1): ");
        Operations.printMatrix(xor.inference(trainInputs[3])[0]);

        System.out.println();
        System.out.println();
        System.out.println();
        for (String s : xor.getNetworkString()){
            System.out.println(s);
        }
    }
}
