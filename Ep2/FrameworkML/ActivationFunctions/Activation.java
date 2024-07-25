package Ep2.FrameworkML.ActivationFunctions;

import Ep2.FrameworkML.Matrix;

/** Interface for the different activation function that can be used in Operations.Java
 * and in the NeuralNet framework. */
public interface Activation {
    /**
     * Takes in a float and returns a float according to that function
     */
    float apply(float x, Matrix[] zValues);
    float derivative(float x, Matrix[] zValues);

    String name();

}