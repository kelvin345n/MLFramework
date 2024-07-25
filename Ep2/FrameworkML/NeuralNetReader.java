package Ep2.FrameworkML;

import Ep2.FrameworkML.ActivationFunctions.*;
import Ep2.FrameworkML.CostFunctions.BinaryCrossEntropy;
import Ep2.FrameworkML.CostFunctions.Cost;
import Ep2.FrameworkML.CostFunctions.MeanSquareError;
import Ep2.FrameworkML.Layers.Dense;
import Ep2.FrameworkML.Layers.Layer;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/** Specifically used to create a neural network from a saved file. */
public class NeuralNetReader {

    /** When called with a valid neural network saved file, return a neural network or
     * errors when the file does not exist or not in the corrext format. */
    public static NeuralNet2 loadNetwork(String fileName){
        try {
            // All the saves done will be from the FrameworkML/Saves directory
            String directoryPath = "C:\\Users\\nkelv\\OneDrive\\Desktop\\jv\\ml_test\\Ep2\\FrameworkML\\Saves";
            FileReader reader = new FileReader(directoryPath + File.separator + fileName);
            Scanner scan = new Scanner(reader);
            int[] inShape = readIntArray(scan.nextLine());
            int[] outShape = readIntArray(scan.nextLine());
            Cost c = readCostFunction(scan.nextLine());
            List<Layer> arch = new ArrayList<>();
            // The first layer will have its weights and biases in index 0 and 1, 2nd layer: index 2, 3...etc
            // Used later to update the parameters in each layer.
            List<Matrix[]> weightBiases = new ArrayList<>();
            while(scan.hasNextLine()){
                Layer layer = readLayer(scan.nextLine(), scan.nextLine(), scan.nextLine(), weightBiases);
                arch.add(layer);
            }
            reader.close();
            scan.close();

            NeuralNet2 nn = new NeuralNet2(inShape, outShape, arch.toArray(new Layer[0]), c);
            // For every layer in the architecture, update the weights and biases.
            for (int i = 0; i < arch.size(); i++){
                Layer curr = arch.get(i);
                curr.setWeights(weightBiases.get(i*2));
                curr.setBiases(weightBiases.get(i*2+1));
            }
            return nn;
        }
        catch(Exception e) {
            e.getStackTrace();
            throw new IllegalArgumentException("File does not exist or error during loading");
        }
    }

    /** Reads in a name of the cost function and returns the corresponding cost function.
     * If the string is not part of the dictionary, null is returned. */
    private static Cost readCostFunction(String cost){
        Cost c;
        switch(cost){
            case "mse":
                c = new MeanSquareError();
                break;
            case "bce":
                c = new BinaryCrossEntropy();
            default:
                throw new IllegalArgumentException("Cost function: " + cost + " not recognized.");
        }
        return c;
    }
    /** Reads in a name of the activation function and returns a new corresponding activation
     * function. If not a valid activation function, null is returned. */
    private static Activation readActFunction(String act){
        Activation func = switch (act) {
            case "relu" -> new Relu();
            case "sigmoud" -> new Sigmoud();
            case "softmax" -> new Softmax();
            case "tanh" -> new Tanh();
            case "linear" -> new Linear();
            default -> throw new IllegalArgumentException("Activation function: " + act + " not recognized.");
        };
        return func;
    }
    /** Reads in a string representation of a matrix and returns a new corresponding matrix.
     * where the str should be in the format:
     *
     * "ROWS COLS ARRAY_PRINTED"
     *
     * */
    public static Matrix readMatrix(String str){
        // String[0] holds rows, [1] holds cols, [2] holds array
        String[] array = str.split("-");
        return new Matrix(Integer.parseInt(array[0]), Integer.parseInt(array[1]), readFloatArray(array[2]));
    }
    /** Reads in a string representation of a layer and returns a new corresponding layer.
     *
     * Dense: Dense NeuronCount ActivationFunction (description)
     *        WEIGHT MATRICES                      (weights)
     *        BIAS MATRIX                          (biases)
     *
     * Conv3D: Description
     *         weight matrices
     *         bias matrices
     * */
    private static Layer readLayer(String description, String weightStr, String biasStr, List<Matrix[]> weightBiases){
        Layer layer;
        // index 0: layerType
        String[] descArray = description.split("-");
        layer = switch (descArray[0]) {
            case "Dense" -> createDense(descArray);
            case "Conv3D" -> createConv3D(descArray);
            default -> throw new IllegalArgumentException("Layer type: " + descArray[0] + " not recognized.");
        };
        String[] weightMats = weightStr.split("~");
        String[] biasMats = biasStr.split("~");
        if (weightMats.length != biasMats.length){
            throw new IllegalArgumentException("the weights and biases should have the same" +
                    "depth");
        }
        Matrix[] weights = new Matrix[weightMats.length];
        Matrix[] biases = new Matrix[biasMats.length];
        for (int i = 0; i < weightMats.length; i++){
            weights[i] = readMatrix(weightMats[i]);
            biases[i] = readMatrix(biasMats[i]);
        }
        assert layer != null;
        weightBiases.add(weights);
        weightBiases.add(biases);
        return layer;
    }

    /** Creates and returns a dense layer based on the set of description given
     *
     * description: Index 0: LayerType, Index1: NeuronCount, Index2: activation func
     *
     * */
    private static Layer createDense(String[] description){
        int neuronCount = Integer.parseInt(description[1]);
        Activation func = readActFunction(description[2]);
        return new Dense(neuronCount, func);
    }
    /** Creates and returns a 3D convolutional layer based on the set of description given */
    private static Layer createConv3D(String[] description){
        // Index 0: LayerType
        return null;
    }

    /** Given a string representation of an integer array, returns a integer array of that string */
    public static int[] readIntArray(String str){
        str = str.substring(1, str.length()-1);
        String[] array = str.split(", ");
        // Step 2: Create an int array to hold the numbers
        int[] numbers = new int[array.length];
        // Step 3: Convert each trimmed string to an int and store in the int array
        for (int i = 0; i < array.length; i++) {
            numbers[i] = Integer.parseInt(array[i].trim());
        }
        return numbers;
    }
    /** Given a string rep of a float array, returns a float array. */
    public static float[] readFloatArray(String str){
        str = str.substring(1, str.length()-1);
        String[] array = str.split(", ");
        // Step 2: Create an int array to hold the numbers
        float[] numbers = new float[array.length];
        // Step 3: Convert each trimmed string to an int and store in the int array
        for (int i = 0; i < array.length; i++) {
            numbers[i] = Float.parseFloat(array[i].trim());
        }
        return numbers;
    }
}
