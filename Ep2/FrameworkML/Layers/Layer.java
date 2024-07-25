package Ep2.FrameworkML.Layers;

import Ep2.FrameworkML.ActivationFunctions.Activation;
import Ep2.FrameworkML.CostFunctions.Cost;
import Ep2.FrameworkML.Matrix;

import java.util.List;

public interface Layer {
    /** Given a layout of what type of input that layer should receive, the layer
     * then uses that information to build its weights and biases. */
    void setInputShape(int[] shape);
    /** Outputs the shape that that layer will output. */
    int[] getOutputShape();

    /** Returns the input shape that this layer expects to receive */
    int[] getInputShape();


    /** Returns a list of weight matrices that correspond to that layer */
    Matrix[] getWeights();
    /** Returns a list of bias matrices that correspond to that layer */
    Matrix[] getBiases();
    Activation getActivationFunc();

    /** Returns the z matrix for that layer. Everytime this function is called, the dj/dz
     * for that layer should have been computed beforehand. */
    Matrix[] getDJ_DZ();

    /** After forward prop and back prop have been called for all layers for all training
     * examples, call this function to update the parameters at that layer. */
    void updateParameters(float learningRate, int trainingCount);

    /** When given a set of inputs, uses matrix multiplication on the weights,
     * adds the bias, and applies the activation function to all elements.
     * Note: Can only be used up to 3 dimensions */
    Matrix[] feedForwardTraining(Matrix[] inputs);

    Matrix[] feedForwardInference(Matrix[] inputs);

    /** Sets the next layer for the current layer's to "next" */
    void setNextLayer(Layer next);

    void backprop(Cost cost, Matrix[] expected);

    /** Returns a string representation of the layer. */
    List<String> stringLayer();

    void setWeights(Matrix[] weight);
    void setBiases(Matrix[] bias);


    /** Checks the given shape if there are any zero's in that shape */
    default void checkShape(int[] shape){
        for (int i = 0; i < shape.length; i++){
            if (shape[i] <= 0){
                throw new IllegalArgumentException("No part of the input shape can be zero or less");
            }
        }
    }

}
