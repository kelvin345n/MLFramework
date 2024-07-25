//package Ep2.FrameworkML;
//
//import Ep2.FrameworkML.ActivationFunctions.Activation;
//import Ep2.FrameworkML.Matrix;
//import net.sf.saxon.expr.Component;
//
///** Framework for creating a neural network. What we want to do is specify for example,
// *
// * 2 inputs, so create a 1 by 2 matrix
// * 2 neurons in the next layer. So create another matrix for that.
// * 1 neuron in the next layer. So create another matrix for that
// * 1 output, so create a matrix for that. for example 1 by 1 matrix
// *
// *
// * Goal: is to give it the framework of a neural network and training data it needs, and it
// * then trains the network and outputs a NeuralNet with the right parameters.
// *
// * */
//public class NeuralNet {
//
//    /**
//     * args:
//     * inputsCount: The number of inputs that the matrix will take in. For example, if we are modeling an AND
//     *                gate, then we will have something like [0 0], [0 1], [1 0] and [1 1] as our input data.
//     *                So the number of inputs should be 2.
//     *
//     * outputsCount: Like the inputs, but for the outputs. For example, the AND gate. We are given 0 and a 0,
//     *                 therefore we must output a 0. In a matrix this output will be represented as [0]. So the number
//     *                 outputs of this neural network is 1.
//     *
//     *
//     * neuronPerLayers: is an int array. The length of layers is how many layers are in
//     *          the neural network. The elements at each index are how many neurons
//     *          are in that layer. For example: [2, 2, 4]. This neural network has 3 hidden layers.
//     *          Note: that the total number of layers is the number of hidden layers + 2 (input and output layers).
//     *          So this example would have 5 total layers.
//     *          The 0th index is the first hidden layer, the last index is the last hidden layer, and everything
//     *          in between are also hidden layers.
//     *
//     *          In the example above, the first hidden layer has 2 neurons, the 2nd hidden layer also has 2 neurons,
//     *          and the last hidden layer has 4 neurons.
//     *
//     *          Note to self: each layer must have its own matrices associated with that layer.
//     *
//     *          The hidden layers however each need a weight matrix and a bias matrix. These will
//     *          have different dimension, so we cannot combine them.
//     *
//     *          The input layer will be just one matrix, where each element is the input.
//     *          The output layer will also just be one matrix.
//     *
//     *          Okay, so we know what size the input layer will be. Let's sa the input layer takes in
//     *          a 1 by 2 matrix. This means it has two inputs. 2 inputs, so the first hidden layer
//     *          each neuron needs two weights one for each of the two inputs x1, x2. If there are n inputs,
//     *          then each neuron inside the first hidden layer needs n weights. x1, x2, x3 ... xn-1, xn.
//     *
//     *          Each neuron in a hidden layer needs a weight matrix and a bias matrix.
//     *
//     * trainingData: This is what the neural net will use to train itself. It should be in the format
//     *              of each row is a separate training input.
//     *              For example the matrix:
//     *
//     *              [1 2 3 4 5]
//     *              [5 4 3 2 1]
//     *              [0 0 0 0 0]
//     *
//     *              This matrix if 3 by 5. therefore it has 3 different training inputs that
//     *              will be put separately through the neural network. The 5 cols are how many
//     *              different inputs such as x1, x2, x3, x4 etc. are put into the neural network.
//     *
//     * expectedData: This is the expected data that the neural network should output when given
//     *               a training input from the training data.
//     *               For example the matrix:
//     *
//     *               [1]
//     *               [2]
//     *               [3]
//     *
//     *               This is a 3 by 1 matrix, therefore when one set of inputs are put in the
//     *               neural net, the output should be one element. And there are 3 different samples.
//     *
//     *               Note: Training data inputs and expected data outputs are synced via their
//     *                     row number. The first row in the training data should output the first
//     *                     row in the expected data when run through the neural network.
//     *
//     * paramLow: When randomly setting the weights and biases, the paramLow is the lowest that
//     *
//     *
//     *
//     * */
//
//    // These will hold all the matrices in the hidden layers.
//    Matrix[] hiddenLayerWeights;
//    Matrix[] hiddenLayerBiases;
//    int hiddenLayerCount;
//    //
//    Matrix trainingData;
//    Matrix expectedData;
//
//    Activation actFunc;
//
//    public NeuralNet(int inputsCount, int outputsCount, int[] neuronPerLayers,
//                     Matrix trainingData, Matrix expectedData,
//                     int paramLow, int paramHigh, Activation function){
//        actFunc = function;
//        hiddenLayerCount = neuronPerLayers.length;
//        if (neuronPerLayers == null || neuronPerLayers.length == 0) {
//            throw new IllegalArgumentException("neuronPerLayers cannot be null or empty.");
//        }
//
//        if (inputsCount <= 0 || outputsCount <= 0){
//            throw new IllegalArgumentException("The input and output count cannot be" +
//                    "zero or less.");
//        }
//        // The number of neurons in the last layer must be the same as the output count
//        if (neuronPerLayers[hiddenLayerCount-1] != outputsCount){
//            throw new IllegalArgumentException("The last hidden layer neuron count " +
//                    "must be the same as the output count");
//        }
//        // The training data must fit the description of input count.
//        if (trainingData.getCols() != inputsCount){
//            throw new IllegalArgumentException("The training data's inputs do not" +
//                    "match the number of inputs expected from the neural net.");
//        }
//        // The expected data must fit the description of output count.
//        if (expectedData.getCols() != outputsCount){
//            throw new IllegalArgumentException("The expected data's output does not " +
//                    "the expected number of outputs from the neural network.");
//        }
//        // Check if the training data and expected data match.
//        if (trainingData.getRows() != expectedData.getRows()){
//            throw new IllegalArgumentException("The amount of training inputs do not match" +
//                    "the number of training outputs.");
//        }
//
//        // The Weights matrix array will hold the weights matrix for that respective hidden layer.
//        // So the first hidden layer will have its weights' matrix in the 0th index. The second
//        // hidden layer will have its weight matrix in the 1st index.
//
//        // The Biases matrix array will be like the Weights, but it holds the bias for that hidden
//        // layer. The bias for the first hid layer will be in the 0th index and so on. The last hidden
//        // layer's bias will be in hiddenLayerCount - 1 index.
//        hiddenLayerWeights = new Matrix[hiddenLayerCount];
//        hiddenLayerBiases = new Matrix[hiddenLayerCount];
//
//        int size = inputsCount;
//        for (int i = 0; i < hiddenLayerCount; i++){
//            // Number of neurons for that hidden layer.
//            int numNeurons = neuronPerLayers[i];
//            // Weights matrix
//            hiddenLayerWeights[i] = new Matrix(size, numNeurons);
//            // Randomize the newly created matrix.
//            hiddenLayerWeights[i].randomizeMatrix(paramLow, paramHigh);
//            // Bias matrix
//            hiddenLayerBiases[i] = new Matrix(1, numNeurons);
//            hiddenLayerBiases[i].randomizeMatrix(paramLow, paramHigh);
//            // Next layer will have its row size be the number of neurons of the prev layer.
//            size = numNeurons;
//        }
//        // Okay so now we have each hidden layer's weights and biases inserted and
//        // randomized.
//        this.trainingData = trainingData;
//        this.expectedData = expectedData;
//    }
//    /** If not given a low and high parameters to set the parameters in the random.
//     * It is automatically set to  */
//    public NeuralNet(int inputsCount, int outputsCount,
//                     int[] neuronPerLayers, Matrix trainingData, Matrix expectedData,
//                     Activation function){
//        this(inputsCount, outputsCount, neuronPerLayers, trainingData, expectedData,
//        0, 1, function);
//    }
//    /** We start training the neural network based on a training set
//     * and expected set, given a learning RATE and EPSILON as many times
//     * as COUNT*/
//    public void trainNeuralNet(int epochs, float rate, float epsilon){
//        // Randomizing the initial parameters in the model
//        int trainingCount = epochs;
//        System.out.println("Init cost: " + cost(actFunc));
//        for (int i = 0; i < trainingCount; i++) {
//            trainNeuralNetOnce(epsilon, rate);
////            if (i % 1000 == 0) { // Log cost every 100 iterations, for example.
////                System.out.println("Trial " + i + " : " + cost(actFunc));
////            }
//        }
//        System.out.println("Final cost: " + cost(actFunc));
//    }
//
//    /** Given a list of Matrices that represent the neural network, where each
//     * is a part of a neural layer.
//     * We train each matrix once, and add its finite difference matrix */
//    private void trainNeuralNetOnce(float eps, float rate){
//        Matrix[] gradientOfWeights = new Matrix[hiddenLayerCount];
//        Matrix[] gradientOfBiases = new Matrix[hiddenLayerCount];
//        // We are finding the finite differences of all the layers first before updating
//        for (int i = 0; i < hiddenLayerWeights.length; i++){
//            Matrix finDiff = finiteDifference(hiddenLayerWeights[i], eps, rate, actFunc);
//            gradientOfWeights[i] = finDiff;
//        }
//        for (int i = 0; i < hiddenLayerBiases.length; i++){
//            Matrix finDiff = finiteDifference(hiddenLayerBiases[i], eps, rate, actFunc);
//            gradientOfBiases[i] = finDiff;
//        }
//        // Once the entire finite difference of the neural network is found, then
//        // we can add to each layer.
//        for (int i = 0; i < hiddenLayerWeights.length; i++){
//            Operations.sumMatrix(hiddenLayerWeights[i], gradientOfWeights[i]);
//        }
//        for (int i = 0; i < hiddenLayerBiases.length; i++){
//            Operations.sumMatrix(hiddenLayerBiases[i], gradientOfBiases[i]);
//        }
//    }
//
//    /** Given a neural layer, we calculate how much to change each parameter
//     *  inside the neural layer in the return matrix. This new matrix will have the same
//     *  dimensions as the neural layer. Now we just need to sum the neural layer and the
//     *  output matrix */
//    private Matrix finiteDifference(Matrix layer, float eps, float rate, Activation function){
//        Matrix output = new Matrix(layer.getRows(), layer.getCols());
//        float temp;
//        float c = cost(function);
//        for (int j = 1; j <= layer.getRows(); j++){
//            for (int k = 1; k <= layer.getCols(); k++){
//                temp = layer.getElement(j, k);
//                // Add epsilon to the current element at this specific layer.
//                layer.setElement(j, k, (temp - eps));
//                // Setting the output to how much to adjust the old matrix
//                output.setElement(j, k, rate*((cost(function) - c) / eps));
//                // Setting the weights
//                layer.setElement(j, k, temp);
//            }
//        }
//        return output;
//    }
//
//    /** calculates the mean squared error (MSE) of training inputs and the
//     * expected value of each training input. */
//    public float cost(Activation function){
//        int n = trainingData.getRows();
//        float cost = 0f;
//        for (int i = 1; i <= n; i++){
//            Matrix output = forward(trainingData.getRowAt(i), function);
//            Matrix expectedMat = expectedData.getRowAt(i);
//            for (int j = 1; j <= output.getCols(); j++){
//                float actual = output.getElement(1, j);
//                float expected = expectedMat.getElement(1, j);
//                cost += (float) Math.pow(actual - expected, 2);
//            }
//        }
//        return cost/n;
//    }
//
//    /** Puts an input matrix through the neural network. Outputs the answer
//     * matrix */
//    private Matrix forward(Matrix input, Activation function){
//        // If the neural net has no hidden layers, output the input
//        Matrix output = input;
//        for (int i = 0; i < hiddenLayerCount; i++){
//            output = goThroughLayer(i, output, function);
//        }
//        return output;
//    }
//
//    /** given which hidden layer to go through, we take the input matrix and go
//     * through that specific hidden layer. This process outputs the next matrix
//     * that should go through the next layer. Keep in mind that this is zero indexed.
//     * the first hidden layer is index 0. The nth hidden layer should be n-1 index. */
//    private Matrix goThroughLayer(int hidLayerIndex, Matrix input, Activation function){
//        if (hidLayerIndex < 0 || hidLayerIndex >= hiddenLayerCount){
//            throw new IllegalArgumentException("The provided index is incompatible" +
//                    "with this neural network. Remember it is zero indexed.");
//        }
//        int i = hidLayerIndex;
//        Matrix weight = hiddenLayerWeights[i];
//        Matrix bias = hiddenLayerBiases[i];
//
//        Matrix output = Operations.dotMatrix(input, weight);
//        Operations.sumMatrix(output, bias);
//        Operations.activationFunction(output, function);
//
//        return output;
//    }
//
//
//    /** Tests the neural network based on the given matrices. Does not train
//     * the network with these matrices. */
//    public void testNeuralNet(Matrix testingInputs){
//        if (testingInputs.getCols() != trainingData.getCols()){
//            throw new IllegalArgumentException("Testing inputs are do not have the right dimensions");
//        }
//        for (int r = 1; r <= testingInputs.getRows(); r++){
//            Matrix input = testingInputs.getRowAt(r);
//            Matrix output = forward(input, actFunc);
//            int count = 1;
//            System.out.println("-------------------------------");
//            System.out.println("INPUT");
//            Operations.printMatrix(input);
//            System.out.println("--------->");
//            System.out.println("OUTPUT");
//            Operations.printMatrix(output);
//            System.out.println("-------------------------------");
//        }
//    }
//
//
//    /** Saves the neural network's parameters into a file*/
//    public void saveNeuralNetwork(){
//
//    }
//
//    /** Loads a neural network from a previously saved one. */
//    public void loadNeuralNetwork(){
//
//    }
//
//    /** Prints into the console, All the layers and the weights and
//     * biases associated with that layer. */
//    public void printNeuralNet(){
//        for (int i = 0; i < hiddenLayerCount; i++){
//            Matrix w = hiddenLayerWeights[i];
//            Matrix b = hiddenLayerBiases[i];
//
//            System.out.println("----------------------------------------");
//            System.out.println("            HIDDEN LAYER " + i);
//            System.out.println("----------------------------------------");
//            System.out.println("WEIGHTS");
//            Operations.printMatrix(w);
//            System.out.println("BIASES");
//            Operations.printMatrix(b);
//        }
//    }
//}
