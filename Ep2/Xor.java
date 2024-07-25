package Ep2;

import Ep2.FrameworkML.*;

public class Xor {



    /** Each row is a different training set. The columns are x1 and x2. */
    Matrix trainingSet = new Matrix(4, 2, new float[]{
            0, 0,
            0, 1,
            1, 0,
            1, 1,
    });
    /** The expected answer for each of the training rows are here. the first row
     * of the training set corresponds to the first row in expected, and so on. */
    Matrix expectedSet = new Matrix(4, 1, new float[]{
            0,
            1,
            1,
            0
    });

    // Weight1 and Bias1
    Matrix w1 = new Matrix(2, 2);
    Matrix b1 = new Matrix(1, 2);

    Matrix w2 = new Matrix(2, 1);
    Matrix b2 = new Matrix(1, 1);

    Matrix[] neuralNetwork = new Matrix[]{w1, b1, w2, b2};

    public Xor(){
        w1.randomizeMatrix(0, 1);
        b1.randomizeMatrix(0, 1);
        w2.randomizeMatrix(0, 1);
        b2.randomizeMatrix(0, 1);
    }


    public void training() {
        // Randomizing the initial parameters in the model
        int trainingCount = 25000;
        float eps = 1e-1f;
        float rate = 1e-1f;
        System.out.println("Init cost: " + cost());
        for (int i = 0; i < trainingCount; i++) {
            trainNeuralNetOnce(neuralNetwork, eps, rate);
            //System.out.println("Trial " + i + " : " + cost());
        }
        System.out.println("Final cost: " + cost());
    }

    /** Given a list of Matrices that represent the neural network, where each
     * is a part of a neural layer.
     * We train each matrix once, and add its finite difference matrix */
    private void trainNeuralNetOnce(Matrix[] neuralNet, float eps, float rate){
        int netLength = neuralNet.length;
        Matrix[] diffOfNeuralNet = new Matrix[netLength];
        // We are finding the finite differences of all the layers first before updating
        for (int i = 0; i < netLength; i++){
            Matrix finDiff = finiteDifference(neuralNet[i], eps, rate);
            diffOfNeuralNet[i] = finDiff;
        }
        // Once the entire finite difference of the neural network is found, then
        // we can add to each layer.
        for (int i = 0; i < netLength; i++){
            Operations.sumMatrix(neuralNet[i], diffOfNeuralNet[i]);
        }
    }

    /** Given a neural layer, we calculate how much to change each parameter
     *  inside the neural layer in the return matrix using finite difference approximation.
     *  This new matrix will have the same dimensions as the neural layer. Now we just need
     *  to sum the neural layer and the output matrix */
    private Matrix finiteDifference(Matrix layer, float eps, float rate){
        Matrix output = new Matrix(layer.getRows(), layer.getCols());
        float temp;
        float c = cost();
        for (int j = 1; j <= layer.getRows(); j++){
            for (int k = 1; k <= layer.getCols(); k++){
                temp = layer.getElement(j, k);
                // Add epsilon to the current element at this specific layer.
                layer.setElement(j, k, (temp - eps));
                // Setting the output to how much to adjust the old matrix
                output.setElement(j, k, rate*((cost() - c) / eps));
                // Setting the weights
                layer.setElement(j, k, temp);
            }
        }
        return output;
    }


    public void printParams(){
        System.out.println("==================================");
        System.out.println("w1");
        Operations.printMatrix(w1);
        System.out.println("==================================");
        System.out.println("b1");
        Operations.printMatrix(b1);
        System.out.println("==================================");
        System.out.println("w2");
        Operations.printMatrix(w2);
        System.out.println("==================================");
        System.out.println("b2");
        Operations.printMatrix(b2);
        System.out.println("==================================");
    }

    /** Testing the model */
    public void testModel(){
        // Testing the output of the xor model
        for (int i = 0; i <= 1; i++){
            for (int j = 0; j <= 1; j++){
                Matrix input = new Matrix(1, 2);
                input.setElement(1, 1, i);
                input.setElement(1, 2, j);
                float output = forward(input).getElement(1, 1);
                System.out.printf("(%d, %d) -> %f%n", i, j, output);
            }
        }
    }

    /** Puts an input matrix through the neural network. The matrix input must be
     * 1 by 2. For example the Matrix [1 0] or [0 1] or [0 0] where the first elem
     * is x1, second elem is x2. Outputs the answer matrix which is 1 by 1.  */
    private Matrix forward(Matrix input){
        // Passing input through the first layer.
        Matrix a1 = Operations.dotMatrix(input, w1);       // Multiplying inputs by weights
        Operations.sumMatrix(a1, b1);                      // Applying bias
        Operations.sigmoudMatrix(a1);                      // Sigmoud the output
        // Passing a1 through the second layer.
        Matrix a2 = Operations.dotMatrix(a1, w2);
        Operations.sumMatrix(a2, b2);
        Operations.sigmoudMatrix(a2);

        return a2;
    }

    /** calculates the mean squared error (MSE) of training inputs and the
     * expected value of each training input. */
    public float cost(){
        if (trainingSet.getRows() != expectedSet.getRows()){
            throw new IllegalArgumentException("The training set and expected set have invalid dimensions");
        }
        int n = trainingSet.getRows();

        float cost = 0f;
        for (int i = 1; i <= n; i++){
            float actual = forward(trainingSet.getRowAt(i)).getElement(1, 1);
            float expected = expectedSet.getElement(i, 1);
            float error = actual - expected;
            cost += error * error;
        }
        cost /= n;
        return cost;
    }



}
