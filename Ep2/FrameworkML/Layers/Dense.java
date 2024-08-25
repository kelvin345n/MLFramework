package Ep2.FrameworkML.Layers;

import Ep2.FrameworkML.ActivationFunctions.Activation;
import Ep2.FrameworkML.CostFunctions.Cost;
import Ep2.FrameworkML.LayerDiff;
import Ep2.FrameworkML.Matrix;
import Ep2.FrameworkML.Operations;

import java.util.ArrayList;
import java.util.List;

public class Dense implements Layer {
    // The running total gradient for all training examples
    Matrix[] runningWeightGrad;
    Matrix[] runningBiasGrad;
    // The gradient for the current training example
    Matrix[] weightGrad;
    Matrix[] biasGrad;
    // The parameters for the neural network.
    Matrix[] weights;
    Matrix[] biases;
    // Will hold the last z output. used for backprop. During backprop, this
    // will hold the part deriv. of cost w/rt that z value.
    Matrix[] z;
    // Will hold the last activation output. used for backprop. During backprop, this
    // will hold the partial derivative of the cost with respect to that activation.
    Matrix[] a;

    Activation act;
    int[] inputShape;
    int neurons;
    // Points to the next layer.
    Layer next;

    public Dense (int neuronCount, Activation func){
        neurons = neuronCount;
        act = func;
        next = null;
    }

    @Override
    public void setInputShape(int[] shape) {
        if (shape.length != 3){
            throw new IllegalArgumentException("The shape must be 3-D");
        }
        if (shape[2] != 1){
            throw new IllegalArgumentException("This layer does not accept examples with depth");
        }
        if (shape[0] != 1){
            throw new IllegalArgumentException("input shape must be 2-Dimensional with one row. [1, x]");
        }
        checkShape(shape);
        inputShape = shape;
        int cols = shape[1];
        // Number of columns in weight matrix is equal to number of neurons.
        // Number of rows in weight matrix is equal to number of columns in input
        weights = new Matrix[]{new Matrix(cols, neurons)};
        weights[0].heInitialization(cols);
        weightGrad = new Matrix[]{new Matrix(cols, neurons)};
        runningWeightGrad = new Matrix[]{new Matrix(cols, neurons)};
        // Bias matrix will be a row vector the size of the number of neurons.
        // Or same shape as the input.
        biases = new Matrix[]{new Matrix(1, neurons)};
        biases[0].randomizeMatrix(0, 1);
        biasGrad = new Matrix[]{new Matrix(1, neurons)};
        runningBiasGrad = new Matrix[]{new Matrix(1, neurons)};
        // Z and A are set to zeros
        z = new Matrix[]{new Matrix(1, neurons)};
        a = new Matrix[]{new Matrix(1, neurons)};
    }
    @Override
    public int[] getOutputShape() {
        // Output is a row vector.
        return new int[]{1, neurons, 1};
    }

    /**
     * Returns the input shape that this layer expects to receive
     */
    @Override
    public int[] getInputShape() {
        return inputShape;
    }

    @Override
    public Matrix[] getDJ_DZ(){
        return z;
    }
    @Override
    public Matrix[] getWeights() {
        return weights;
    }
    @Override
    public Matrix[] getBiases() {
        return biases;
    }
    @Override
    public Activation getActivationFunc() {
        return act;
    }

    @Override
    public void updateParameters(float learningRate, int trainingCount){
        updateWeights(learningRate, trainingCount);
        updateBiases(learningRate, trainingCount);
    }

    /** Subtracts from each weight, the average computed gradient for that weight times
     * the learning rate. The updated weights will be in the weights matrix of that layer. */
    private void updateWeights(float learningRate, int trainingCount) {
        Matrix weight = weights[0];
        for (int r = 1; r <= weight.getRows(); r++){
            for (int c = 1; c <= weight.getCols(); c++){
                float dj_dw = runningWeightGrad[0].getElement(r, c)/trainingCount;
                float weightUpdate = weight.getElement(r, c) - learningRate*dj_dw;
                weight.setElement(r, c, weightUpdate);
            }
        }
        // Reset the running weight gradient to 0
        runningWeightGrad = new Matrix[]{new Matrix(inputShape[1], neurons)};
    }
    private void updateBiases(float learningRate, int trainingCount) {
        Matrix bias = biases[0];
        for (int r = 1; r <= bias.getRows(); r++){
            for (int c = 1; c <= bias.getCols(); c++){
                float dj_db = runningBiasGrad[0].getElement(r, c)/trainingCount;
                float biasUpdate = bias.getElement(r, c) - learningRate*dj_db;
                bias.setElement(r, c, biasUpdate);
            }
        }
        runningBiasGrad = new Matrix[]{new Matrix(1, neurons)};
    }

    @Override
    public Matrix[] feedForwardTraining(Matrix[] input){
        // When we feed forward we should set each input the weight is being multiplied by
        // in the same place in the weight gradient. useful later for backprop.
        for (int r = 1; r <= weightGrad[0].getRows(); r++){
            float in = input[0].getElement(1, r);
            for (int c = 1; c <= weightGrad[0].getCols(); c++){
                weightGrad[0].setElement(r, c, in);
            }
        }
        Matrix out = Operations.dotMatrix(input[0], weights[0]);
        Operations.sumMatrix(out, biases[0]);
        // Now we have z, we also need to set it to the z matrix for backprop later.
        z[0] = Operations.copy(out);
        Matrix[] zCopy = new Matrix[]{Operations.copy(out)};
        Operations.activationFunction(out, act, zCopy);
        // Apply activation function to all elements of z and copy it the "a" matrix.
        a[0] = Operations.copy(out);
        return new Matrix[]{out};
    }

    @Override
    public Matrix[] feedForwardInference(Matrix[] input){
        Matrix out = Operations.dotMatrix(input[0], weights[0]);
        Operations.sumMatrix(out, biases[0]);
        Matrix[] zCopy = new Matrix[]{Operations.copy(out)};
        Operations.activationFunction(out, act, zCopy);
        return new Matrix[]{out};
    }

    /**
     * Sets the next layer for the current layer's to "next"
     */
    @Override
    public void setNextLayer(Layer next) {
        this.next = next;
    }

    /** When called assumes the layer in front has been backpropagated, and changes
     * "a" to dj/da, and "z" to dj/dz*/
    public void backprop(Cost cost, Matrix[] expected){
        deriveA(cost, expected);
        deriveZ();
        deriveW();
        deriveB();
    }

    /**
     * Returns a string representation of the layer.
     *
     * Output:
     *
     * First Line: Dense-NEURONCOUNT-ACTFUNCTION
     * Second Line: "list of weight matrices"
     * Third line: "list of bias matrices"
     *
     * So every new matrix on the same line would mean another element to the matrix list or 3-D.
     */
    @Override
    public List<String> stringLayer() {
        List<String> layerRep = new ArrayList<>();
        layerRep.add("Dense<>" + neurons + "<>" + act.name());
        // The next layer holds all the weight matrices all in one line.
        StringBuilder wb = new StringBuilder();
        for (int i = 0; i < weights.length; i++){
            wb.append(Operations.stringMatrix(weights[i]));
            // The "~" will be used to separate each matrix.
            wb.append("~");
        }
        layerRep.add(wb.toString());

        StringBuilder bb = new StringBuilder();
        for (int i = 0; i < biases.length; i++){
            bb.append(Operations.stringMatrix(biases[i]));
            bb.append("~");
        }
        layerRep.add(bb.toString());
        return layerRep;
    }

    /** Gets the derivative of J w/rt each activation value in this layer and
     * sets it to the "a" matrix*/
    private void deriveA(Cost cost, Matrix[] expected){
        if (next == null){
            // If no layer is next that means we are at the output layer. Where we will
            // need to compute dj/da given the cost function as the activation in the output layer
            // is the actual output of the neural network.
            for (int c = 1; c <= a[0].getCols(); c++){
                float yc = expected[0].getElement(1, c);
                float ac = a[0].getElement(1, c);
                float dj_dac = cost.derivative(ac, yc);
                a[0].setElement(1, c, dj_dac);
            }
        } else {
            // Sets the tensor a to a tensor with its dj/da values
            a = LayerDiff.actDiff(a, next);
        }
    }

    /** Computes the derivative of J w/rt to z using the activation vector.
     * And z now represents dj/dz */
    private void deriveZ(){
        Matrix[] zCopy = new Matrix[]{Operations.copy(z[0])};
        for (int c = 1; c <= z[0].getCols(); c++){
            // da/dz * dj/da
            float dj_dz_c = act.derivative(z[0].getElement(1, c), zCopy) * a[0].getElement(1, c);
            z[0].setElement(1, c, dj_dz_c);
        }
    }

    /** dj/dw = the activation that weight was multiplied by times dj/dz for that weight */
    private void deriveW(){
        // We compute the gradient of each weight and put it into the weightGrad matrix

        // Assuming the input each weight is being multiplied by is already in weightGrad...
        // We just multiply each column by the column in dj/dz
        // Also assuming dj/dz has been computed.

        for (int r = 1; r <= weightGrad[0].getRows(); r++){
            for (int c = 1; c <= weightGrad[0].getCols(); c++){
                float input = weightGrad[0].getElement(r, c);
                float dj_dzc = z[0].getElement(1, c);
                weightGrad[0].setElement(r, c, input*dj_dzc);
            }
        }
        // Add the computed gradient for the weights for this current training example
        // to the running total for all weight gradients computed for that specific weight.
        Operations.sumMatrix(runningWeightGrad[0], weightGrad[0]);
    }
    /** dj/db is the same as dj/dz because dz/db = 1 */
    private void deriveB(){
        biasGrad = z;
        // Add the computed gradient for the bias to the running total for all
        // gradients computed for that bias.
        Operations.sumMatrix(runningBiasGrad[0], biasGrad[0]);
    }

    /** Useful for only when we want to set our own weights and biases. Used for
     * loading neural networks. */
    @Override
    public void setWeights(Matrix[] weight){
        weights = weight;
    }
    @Override
    public void setBiases(Matrix[] bias){
        biases = bias;
    }
}
