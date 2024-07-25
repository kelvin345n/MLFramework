package Ep2.FrameworkML;
import java.io.*;
import Ep2.FrameworkML.CostFunctions.Cost;
import Ep2.FrameworkML.Layers.Layer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/** This neural network framework can only work up to 3-Dimensions
 * */
public class NeuralNet2 {
    int[] inputShape;
    int[] outputShape;
    Cost cost;
    Layer[] architecture;
    /** inputShape: Can only work up to 3 dimensions. [rows, cols, depth], 2-Dimensions,
     *              Note: 1-Dimensions must be modelled as [1, cols]
     * outputShape: Can work up to 3-dimensions. [rows, cols, depth]
     * architexture: List of all the layers used in the architecture.
     * */
    public NeuralNet2(int[] inputShape, int[] outputShape, Layer[] architecture,
                      Cost cost){
        if (architecture.length == 0){
            throw new IllegalArgumentException("An architecture cannot have zero layers");
        }
        checkShape(inputShape);
        checkShape(outputShape);
        this.inputShape = inputShape;
        this.outputShape = outputShape;
        this.cost = cost;

        int[] input = inputShape;
        Layer prev = null;
        for (Layer currLayer : architecture) {
            // Now time to build the architecture.
            currLayer.setInputShape(input);
            // the current layer's output shape will be the next layer's input shape
            input = currLayer.getOutputShape();

            if (prev != null) {
                prev.setNextLayer(currLayer);
            }
            prev = currLayer;
        }
        // input here is the output of the last layer (output of the network)
        if (!Arrays.equals(input, outputShape)){
            throw new IllegalArgumentException("In compatible outputs. The last layer in " +
                    "the architecture should output the same shape as the given output shape");
        }
        this.architecture = architecture;
    }

    /** Checks if the given shape is valid. */
    private void checkShape(int[] shape){
        if (shape.length != 3){
            throw new IllegalArgumentException("The shape must be 3-Dimension");
        }
        // Checks if any of the dimensions of the shape are set to zero
        for (int j : shape) {
            if (j <= 0) {
                throw new IllegalArgumentException("No part of the input shape can be zero");
            }
        }
    }

    /** Given input (x1, x2,..., xn), outputs a matrix of the inference. */
    public Matrix[] inference(Matrix[] input){
        for (Layer layer : architecture) {
            input = layer.feedForwardInference(input);
        }
        return input;
    }

    /** Computed the cost function given a training example's input and
     * expected output */
    public float cost(Matrix[][] inputs, Matrix[][] expectedes){
        if (inputs.length != expectedes.length){
            throw new IllegalArgumentException("# of training examples for inputs" +
                    "and expected must be the same");
        }
        int m = inputs.length;
        float totalCost = 0;
        for (int i = 0; i < m; i++){
            Matrix[] output = inference(inputs[i]);
            Matrix[] expected = expectedes[i];
            if (output.length != expected.length ||
                    output[0].getRows() != expected[0].getRows() ||
                    output[0].getCols() != expected[0].getCols()){
                throw new IllegalArgumentException("Expected shape does not match the output shape.");
            }
            int depth = output.length;
            int rows = output[0].getRows();
            int cols = output[0].getCols();

            for (int d = 0; d < depth; d++){
                for (int r = 1; r <= rows; r++){
                    for (int c = 1; c <= cols; c++){
                        float out_drc = output[d].getElement(r, c);
                        float exp_drc = expected[d].getElement(r, c);
                        totalCost += cost.cost(out_drc, exp_drc);
                    }
                }
            }
        }
        return totalCost/m;
    }


    /** Given a training set and expected set, where one example is represented by a
     * list of matrices. And the whole training data is a list of these list of matrices
     * be one training example.
     * We train the neural network and update parameters once from this data. */
    public void trainNetwork(Matrix[][] trainingData, Matrix[][] expectedData, float learningRate){
        if (trainingData.length != expectedData.length){
            throw new IllegalArgumentException("The training set and expected set " +
                    "should have the same number of examples");
        }
        if (trainingData.length == 0){
            throw new IllegalArgumentException("Must have minimum one training example");
        }
        int m = trainingData.length;
        for (int i = 0; i < m; i++){
            Matrix[] example = trainingData[i];
            Matrix[] expected = expectedData[i];
            checkExample(example, expected);
            // Okay example and expected can be used in the neural network.
            // Forward prop
            for (Layer layer : architecture) {
                example = layer.feedForwardTraining(example);
            }
            // backprop
            for (int j = architecture.length-1; j >= 0; j--){
                architecture[j].backprop(cost, expected);
            }
        }
        for (Layer layer : architecture) {
            layer.updateParameters(learningRate, m);
        }
    }
    /** Checks if the shapes of the given training example and corresponding expected
     * matches the shape for the neural network. */
    private void checkExample(Matrix[] trainExample, Matrix[] expectExample){
        int trDepth = trainExample.length;
        int exDepth = expectExample.length;
        if (trDepth == 0 || exDepth == 0){
            throw new IllegalArgumentException("The training example cannot be empty");
        }
        int[] trainShape = new int[]{trainExample[0].getRows(), trainExample[0].getCols(), trDepth};
        int[] expectShape = new int[]{expectExample[0].getRows(), expectExample[0].getCols(), exDepth};
        if (!Arrays.equals(trainShape, inputShape)){
            throw new IllegalArgumentException("The training example input size is not compatible with" +
                    "the neural network.");
        }
        if (!Arrays.equals(expectShape, outputShape)){
            throw new IllegalArgumentException("The expected example output size is not compatible with" +
                    "the neural network");
        }
    }

    /** Saves the structure and all the parameters of the neural network. */
    public void saveNetwork(String fileName){
        try {
            // Specify the directory path
            String directoryPath = "C:\\Users\\nkelv\\OneDrive\\Desktop\\jv\\ml_test\\Ep2\\FrameworkML\\Saves";
            // Create a File object representing the new file
            File file = new File(directoryPath + File.separator + fileName);

            // Create the file
            if (file.createNewFile()) {
                PrintWriter pw = new PrintWriter(file);
                for (String line : getNetworkString()) {
                    pw.println(line);
                }
                // Write neural network parameters into file.
                pw.close();
                System.out.println("Save successful to " + file.getName() + ".txt");
            } else {
                System.out.println("File already exists. Do you want to overwrite (Y/N)?");
                Scanner scan = new Scanner(System.in);  // Create a Scanner object
                boolean t = true;
                while (t){
                    String overwrite = scan.nextLine();  // Read user input
                    switch(overwrite.toLowerCase()){
                        case "y":
                            PrintWriter pw = new PrintWriter(file);
                            for (String line : getNetworkString()){
                                pw.println(line);
                            }
                            // Write neural network parameters into file.
                            pw.close();
                            System.out.println("Overwrite successful to " + file.getName() + ".txt");
                            t = false;
                            break;
                        case "n":
                            System.out.println("Save unsuccessful.");
                            t = false;
                            break;
                        default:
                            System.out.println("Unknown response. Try again (y/n)");
                            break;
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /** Returns a list of strings that need to be written into the save file line by line. */
    public List<String> getNetworkString(){
        List<String> net = new ArrayList<>();
        // First line is the input shape. For example inputShape = {4, 5, 1} would be displayed as 4 5 1
        net.add(Arrays.toString(inputShape));
        // Second line is the output shape.
        net.add(Arrays.toString(outputShape));
        // Third line is the cost function.
        net.add(cost.name());
        // Listing out the architecture. Every 3 lines is the structure of a layer.
        for (Layer l : architecture) {
            net.addAll(l.stringLayer());
            // This adds the type of layer and all info needed to create that layer
            // And all the weights and bias matrices
        }
        return net;
    }
}
