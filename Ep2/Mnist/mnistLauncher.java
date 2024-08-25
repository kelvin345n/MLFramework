package Ep2.Mnist;

import Ep2.FrameworkML.ActivationFunctions.Relu;
import Ep2.FrameworkML.ActivationFunctions.Softmax;
import Ep2.FrameworkML.CostFunctions.MeanSquareError;
import Ep2.FrameworkML.Layers.Dense;
import Ep2.FrameworkML.Layers.Layer;
import Ep2.FrameworkML.Matrix;
import Ep2.FrameworkML.NeuralNet2;
import Ep2.FrameworkML.NeuralNetReader;
import Ep2.FrameworkML.Operations;

import java.util.List;

public class mnistLauncher {
    public static void main(String[] args){
        List<List<Matrix[]>> trainData = loadData.load("D:/mnist dataset/mnist/mnist_train.csv");
        Matrix[][] trainInputs = trainData.getFirst().toArray(new Matrix[0][]);
        Matrix[][] trainOutputs = trainData.getLast().toArray(new Matrix[0][]);

        List<List<Matrix[]>> testData = loadData.load("D:/mnist dataset/mnist/mnist_test.csv");
        Matrix[][] testInputs = testData.getFirst().toArray(new Matrix[0][]);
        Matrix[][] testOutputs = testData.getLast().toArray(new Matrix[0][]);

        int[] inputShape = new int[]{1, 784, 1};
        int[] outputShape = new int[]{1, 10, 1};
        Layer[] arch = new Layer[]{
                new Dense(30, new Relu()),
                new Dense(10, new Softmax())
        };
        NeuralNet2 nn = new NeuralNet2(inputShape, outputShape, arch, new MeanSquareError());
        System.out.println("Initial accuracy");
        float acc = testAccuracy(nn, testInputs, testOutputs);
        int epochs = 60;
        float learningRate = 0.01f;
        for (int i = 1; i <= epochs; i++){
            nn.trainNetwork(trainInputs, trainOutputs, learningRate);
            System.out.println("Epoch " + i + " accuracy");
            float acc1 = testAccuracy(nn, testInputs, testOutputs);
            if (acc1 < acc){
                learningRate -= 0.0005f;
                System.out.println(learningRate);
            }
            acc = acc1;
        }
        nn.saveNetwork("number");
    }

    /** Given a testing set, returns the floating point accuracy of the neural network inferences */
    public static float testAccuracy(NeuralNet2 nn, Matrix[][] testInputs, Matrix[][] testOutputs){
        if (testInputs.length != testOutputs.length){
            throw new IllegalArgumentException("Input length must be equal to output length");
        }
        int correct = 0;
        for (int i = 0; i < testInputs.length; i++){
            Matrix[] infer = nn.inference(testInputs[i]);
            int inferInteger = predictNumber(infer);
            int actualInteger = predictNumber(testOutputs[i]);
            if (inferInteger == actualInteger){
                correct++;
            }
        }
        System.out.println(correct + "/" + testInputs.length);
        return (float) correct/testInputs.length;
    }

    /** Given an array of matrices, predicts the number that has the highest probability from
     * inference matrix*/
    public static int predictNumber(Matrix[] inference){
        float highest = 0;
        int index = 0;
        Matrix m = inference[0];
        float[] values = m.getMatrixArray();
        for (int i = 0; i < values.length; i++){
            if (Float.isNaN(values[i])){
                throw new IllegalArgumentException("Not a Number");
            }
            if (values[i] > highest){
                highest = values[i];
                index = i;
            }
        }
        return index;
    }

}
