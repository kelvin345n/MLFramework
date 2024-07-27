package Ep2.FrameworkML.Layers;

import Ep2.FrameworkML.ActivationFunctions.Activation;
import Ep2.FrameworkML.CostFunctions.Cost;
import Ep2.FrameworkML.LayerDiff;
import Ep2.FrameworkML.Matrix;
import Ep2.FrameworkML.Operations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Conv3D implements Layer{
    int units;
    // The weights matrix will be split up with the first matrix being all the weights for the first
    // matrix input, the second weight matrix will be for the second matrix input.
    Matrix[] weights;
    Matrix[] biases;
    // Current dj/dw and dj/db for the parameters in this layer
    Matrix[] weightGrad;
    Matrix[] biasGrad;
    // Running total for dj/dw and dj/db for this layer. Still need to divide by training count.
    Matrix[] runningWeightGrad;
    Matrix[] runningBiasGrad;
    // Will hold the last z output. used for backprop. During backprop, each element
    // will hold the partial deriv. of cost w/rt that z value.
    Matrix[] z;
    // Will hold the last activation output. used for backprop. During backprop, each element
    // will be replaced with its partial derivative of the cost with respect to that activation.
    Matrix[] a;

    // We do not need to specify units, because that will be computed when we know the input size
    // the filter size, and the stride between each filter.
    // inputSize = array of that shape of that 3D input. ex: [width, height, depth]
    //            Where width = # of columns, height = # of rows, depth = # of matrices
    int[] inputShape;
    // The output shape should always be three dimensions.
    // The output shape is the same whether flatten is true or not. We need this so
    // we can unflatten the activations of the next layer for backprop.
    int[] outputShape;
    // Activation function that is applied to
    Activation act;

    // The amount to stride each row for ech filter
    int rowStride;
    // The column stride that is used to move each filter.
    int colStride;
    // The depth stride used to move in between each filter.
    int depthStride;

    // How large the filter should be row wise
    int rowFilterSize;
    // The filter size, column wise
    int colFilterSize;
    // How large the filter will be depth wise.
    int depthFilterSize;

    // The number of filters that are used on a single depth of the input, column wise
    int numOfColFilters;
    // The number of filters than are used row wise on a single depth of the input
    int numOfRowFilters;
    // Number of filters that are used depth wise on the inputs
    int numOfDepthFilters;

    // The number of total filters needed to be used on a single depth of the input.
    int filterCount;

    boolean flatten;


    // Next layer in the architecture
    Layer next;

    /**
     * Stride: how much the filter should move in between each convolution. Keep in mind that the depth stride
     *          is going to be the depth of the input shape. I am lazy.
     * Filter size: How big that filter is.
     * Flatten: whether once we are done convolutionizing, if the output should be flattened to a 2-D, row vector.
     *
     * */
    public Conv3D(int rowFilterSize, int colFilterSize, int depthFilterSize, int rowStride, int colStride, int depthStride,
                  Activation func, boolean flatten){
        if (rowFilterSize * colFilterSize * depthFilterSize <= 0){
            throw new IllegalArgumentException("The filter must have row, col, and depth to it.");
        }
        if (rowStride * colStride * depthStride <= 0){
            throw new IllegalArgumentException("The stride values cannot be zero or negative.");
        }
        this.rowFilterSize = rowFilterSize;
        this.colFilterSize = colFilterSize;
        this.depthFilterSize = depthFilterSize;
        this.rowStride = rowStride;
        this.colStride = colStride;
        this.depthStride = depthStride;

        this.flatten = flatten;
        act = func;
        next = null;
    }
    /** If flatten is not specified, default to false */
    public Conv3D(int rowFilterSize, int colFilterSize, int depthFilterSize, int rowStride, int colStride, int depthStride,
                  Activation func){
        this(rowFilterSize, colFilterSize, depthFilterSize, rowStride, colStride, depthStride, func,false);
    }

    @Override
    public void setInputShape(int[] shape) {
        if (shape.length != 3){
            throw new IllegalArgumentException("The input size must be 3-Dimensional");
        }
        checkShape(shape);
        inputShape = shape;
        // To determine the number of neurons in this layer, we figure out how many filters are applied
        // to the input.
        int rows = inputShape[0];
        int cols = inputShape[1];
        int depth = inputShape[2];

        if (colFilterSize > cols || colStride > cols){
            throw new IllegalArgumentException("The column filter size or column stride cannot be" +
                    "greater than the number of columns");
        }
        if (rowFilterSize > rows || rowStride > rows){
            throw new IllegalArgumentException("The row filter size or row stride cannot be" +
                    "greater than the number of row");
        }
        if (depthFilterSize > depth || depthStride > depth){
            throw new IllegalArgumentException("The depth filter size or depth stride cannot be" +
                    "greater than the depth");
        }

        numOfColFilters = (cols - colFilterSize)/colStride + 1;
        numOfRowFilters = (rows - rowFilterSize)/rowStride + 1;
        numOfDepthFilters = (depth - depthFilterSize)/depthStride + 1;

        filterCount = numOfColFilters * numOfRowFilters * numOfDepthFilters;
        // Since each filter outputs one element then numOfColFilters = columns,
        // numOfRowFilters = rows, depth = depth
        outputShape = new int[]{numOfRowFilters, numOfColFilters, numOfDepthFilters};
        // Initializing parameters and helper parameters.
        weights = new Matrix[depthFilterSize];
        weightGrad = new Matrix[depthFilterSize];
        runningWeightGrad = new Matrix[depthFilterSize];

        biases = new Matrix[numOfDepthFilters];
        biasGrad = new Matrix[numOfDepthFilters];
        runningBiasGrad = new Matrix[numOfDepthFilters];

        z = new Matrix[numOfDepthFilters];
        a = new Matrix[numOfDepthFilters];

        for (int i = 0; i < depthFilterSize; i++){
            // Each of the weights will correspond to a matrix that is
            // the filter size x number of filters needed.
            weights[i] = new Matrix(rowFilterSize, colFilterSize * filterCount);
            weights[i].randomizeMatrix(0, 1);
            weightGrad[i] = new Matrix(rowFilterSize, colFilterSize * filterCount);
            runningWeightGrad[i] = new Matrix(rowFilterSize, colFilterSize * filterCount);
        }
        for (int i = 0; i < numOfDepthFilters; i++) {
            // Each neuron will be its filter size and all the weights matrices at the same
            // filter position and size.
            biases[i] = new Matrix(numOfRowFilters, numOfColFilters);
            biases[i].randomizeMatrix(0, 1);
            biasGrad[i] = new Matrix(numOfRowFilters, numOfColFilters);
            runningBiasGrad[i] = new Matrix(numOfRowFilters, numOfColFilters);
            // Each element in these initially set to zero.
            z[i] = new Matrix(numOfRowFilters, numOfColFilters);
            a[i] = new Matrix(numOfRowFilters, numOfColFilters);
        }

    }

    @Override
    public int[] getOutputShape() {
        if (flatten) {
            return new int[]{1, outputShape[0] * outputShape[1] * outputShape[2], 1};
        }
        return outputShape;
    }
    /**
     * Returns the input shape that this layer expects to receive
     */
    @Override
    public int[] getInputShape() {
        return inputShape;
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
    public Matrix[] getDJ_DZ() {
        return z;
    }
    /**
     * Sets the next layer for the current layer's to "next"
     */
    @Override
    public void setNextLayer(Layer next) {
        this.next = next;
    }

    /**
     * When given a set of inputs, uses matrix multiplication on the weights,
     * adds the bias, and applies the activation function to all elements.
     * Note: Can only be used up to 3 dimensions
     *
     * @param inputs
     */
    @Override
    public Matrix[] feedForwardInference(Matrix[] inputs) {
        if (inputs[0].getRows() != inputShape[0] || inputs[0].getCols() != inputShape[1] ||
                inputs.length != inputShape[2]
        ){
            throw new IllegalArgumentException("The given input shape is not" +
                    "compatible with expected input shape");
        }
        // Okay the inputs have the expected size... Now what do I do now?

        // Apply the 3-D filters on each part of the input.

        // Remember that the input, the weights, and the biases have the same depth.
        // Apply each numOfDepthFilter wise.
        // 0 b/c array of matrices.

        // The output should be shape (numOfRowFilters, numOfColFilters, numOfDepthFilters)
        int inputDepthStart = 0;
        // Start at the first filter. filter row start always stays 1.
        // Filter depth start always stays at 0.
        int filterRowStart = 1;
        int filterColStart = 1;
        int filterDepthStart = 0;
        for (int d = 0; d < numOfDepthFilters; d++){
            // We need to apply filter starting at rowStart and colStart.
            int inputRowStart = 1;
            // Apply the filter to each sub tensor of the input.
            for (int r = 1; r <= numOfRowFilters; r++){
                int inputColStart = 1;
                for (int c = 1; c <= numOfColFilters; c++){
                    Matrix[] subInput = Operations.createSubTensor(inputs,
                            inputRowStart, inputColStart, inputDepthStart,
                            rowFilterSize, colFilterSize, depthFilterSize
                    );
                    Matrix[] filter = Operations.createSubTensor(weights,
                            filterRowStart, filterColStart, filterDepthStart,
                            rowFilterSize, colFilterSize, depthFilterSize
                    );
                    float scalar = 0f;
                    // Now multiply them together element wise and add them up.
                    for (int k = 0; k < subInput.length; k++){
                        Matrix prod = Operations.hadamard(subInput[k], filter[k]);
                        scalar += Operations.elementSum(prod);
                    }
                    // Keep in mind bias and z have the same shape.
                    float z_rcd = scalar + biases[d].getElement(r, c);
                    z[d].setElement(r, c, z_rcd);
                    // the filter size stays the same after each operation.
                    // Update the column start for both
                    filterColStart += colFilterSize;
                    inputColStart += colStride;
                }
                // Keep in mind we do not need to check if we run out of bounds
                // with inputRowStart because we checked already how many row filters
                // are used in numOfRowFilters
                inputRowStart += rowStride;
            }
            inputDepthStart += depthStride;
        }
        // Applying activation function to all elements of z
        for (Matrix matrix : z) {
            // When we reach this stage then z[d] has all been filled up with z values.
            // If we were training then we would copy all these values into "a" and then
            // do the activation function on "a". And also return a, or flatten a.
            Operations.activationFunction(matrix, act, z);
        }
        if (flatten){
            return Operations.flatten(z);
        }
        return z;
    }
    @Override
    public Matrix[] feedForwardTraining(Matrix[] inputs) {
        // Remember to save z and a. And set inputs into weightGrad
        if (inputs[0].getRows() != inputShape[0] || inputs[0].getCols() != inputShape[1] ||
                inputs.length != inputShape[2]
        ){
            throw new IllegalArgumentException("The given input shape is not" +
                    "compatible with expected input shape");
        }
        // The output should be shape (numOfRowFilters, numOfColFilters, numOfDepthFilters)
        int inputDepthStart = 0;
        // Start at the first filter. filter row start always stays 1.
        // Filter depth start always stays at 0.
        int filterRowStart = 1;
        int filterColStart = 1;
        int filterDepthStart = 0;
        for (int d = 0; d < numOfDepthFilters; d++){
            // We need to apply filter starting at rowStart and colStart.
            int inputRowStart = 1;
            // Apply the filter to each sub tensor of the input.
            for (int r = 1; r <= numOfRowFilters; r++){
                int inputColStart = 1;
                for (int c = 1; c <= numOfColFilters; c++){
                    Matrix[] subInput = Operations.createSubTensor(inputs,
                            inputRowStart, inputColStart, inputDepthStart,
                            rowFilterSize, colFilterSize, depthFilterSize
                    );
                    Matrix[] filter = Operations.createSubTensor(weights,
                            filterRowStart, filterColStart, filterDepthStart,
                            rowFilterSize, colFilterSize, depthFilterSize
                    );
                    float scalar = 0f;
                    // Now multiply them together element wise and add them up.
                    for (int k = 0; k < subInput.length; k++){
                        Matrix prod = Operations.hadamard(subInput[k], filter[k]);
                        scalar += Operations.elementSum(prod);
                    }
                    // We need to set each weight's input it received to weightGrad
                    Operations.addSubTensor(weightGrad, subInput, filterRowStart, filterColStart, filterDepthStart);

                    // Keep in mind bias and z have the same shape.
                    float z_rcd = scalar + biases[d].getElement(r, c);
                    z[d].setElement(r, c, z_rcd);
                    // the filter size stays the same after each operation.
                    // Update the column start for both
                    filterColStart += colFilterSize;
                    inputColStart += colStride;
                }
                // Keep in mind we do not need to check if we run out of bounds
                // with inputRowStart because we checked already how many row filters
                // are used in numOfRowFilters
                inputRowStart += rowStride;
            }
            inputDepthStart += depthStride;
        }
        // Copying all elements of z into a, one matrix at a time. a and z should have the sam dimensions.
        for (int d = 0; d < a.length; d++) {
            a[d] = Operations.copy(z[d]);
        }
        for (Matrix m : a){
            // When we reach this stage then a has all been filled up with z values.
            // So we apply the activation function to all elements
            Operations.activationFunction(m, act, a);
        }
        if (flatten){
            return Operations.flatten(a);
        }
        return a;
    }

    @Override
    public void backprop(Cost cost, Matrix[] expected) {
        deriveA(cost, expected);
        deriveZ();
        deriveW();
        deriveB();
    }
    /** Gets the derivative of J w/rt each activation value in this layer and
     * sets it to the "a" matrix*/
    private void deriveA(Cost cost, Matrix[] expected){
        if (next == null){
            // If no layer is next that means we are at the output layer. Where we will
            // need to compute dj/da given the cost function as the activation in the output layer
            // is the actual output of the neural network.

            int[] expShape = new int[]{expected[0].getRows(), expected[0].getCols(), expected.length};

            if (!Arrays.equals(outputShape, expShape)){
                // This means that the output was reshaped before being output.
                expected = Operations.reshape(expected, outputShape);
            }
            for (int d = 0; d < a.length; d++){
                for (int r = 1; r <= a[0].getRows(); r++){
                    for (int c = 1; c <= a[0].getCols(); c++){
                        float yc = expected[d].getElement(r, c);
                        float ac = a[d].getElement(r, c);
                        float dj_dac = cost.derivative(ac, yc);
                        a[d].setElement(r, c, dj_dac);
                    }
                }
            }
        } else {
            a = LayerDiff.actDiff(a, next);
        }
    }

    /** Computes the derivative of J w/rt to z using the dj/da tensor.
     * And z now represents dj/dz */
    private void deriveZ(){
        // Copy to a new tensor because we want the original values to send into the
        // activation function later. Think softmax function.
        Matrix[] zCopy = new Matrix[z.length];
        for (int d = 0; d < zCopy.length; d++){
            zCopy[d] = Operations.copy(z[d]);
        }

        for (int d = 0; d < z.length; d++){
            for (int r = 1; r <= z[0].getRows(); r++){
                for (int c = 1; c <= z[0].getCols(); c++){
                    // da/dz * dj/da
                    float dj_dz_rcd = act.derivative(z[d].getElement(r, c), zCopy) * a[d].getElement(r, c);
                    z[d].setElement(r, c, dj_dz_rcd);
                }
            }
        }
    }

    /** dj/dw = the activation that weight was multiplied by times dj/dz for that weight */
    private void deriveW(){
        int colFilterStart = 1;
        // We add colFilterSize to colFilterStart for each filter.
        for (int d = 0; d < z.length; d++){
            for (int r = 1; r <= z[0].getRows(); r++){
                for (int c = 1; c <= z[0].getCols(); c++){
                    float dj_dz = z[d].getElement(r, c);
                    // We multiply the dj/dz value to each element in the associated filter.
                    for (int d_w = 0; d_w < weightGrad.length; d_w++){
                        for (int r_w = 1; r_w <= weightGrad[0].getRows(); r_w++){
                            for (int c_w = 0; c_w < colFilterSize; c_w++){
                                float input = weightGrad[d_w].getElement(r_w, colFilterStart+c_w);
                                weightGrad[d_w].setElement(r_w, colFilterStart+c_w, input*dj_dz);
                            }
                        }
                    }
                    // Increment to next filter for each z value.
                    colFilterStart += colFilterSize;
                }
            }
        }
        // Add the computed gradient for the weights for this current training example
        // to the running total for all weight gradients computed for that specific weight.
        for (int d = 0; d < runningWeightGrad.length; d++){
            Operations.sumMatrix(runningWeightGrad[d], weightGrad[d]);
        }
    }
    /** dj/db is the same as dj/dz because dz/db = 1 */
    private void deriveB(){
        biasGrad = z;
        // Add the computed gradient for the bias to the running total for all
        // gradients computed for that bias.
        for (int d = 0; d < runningBiasGrad.length; d++){
            Operations.sumMatrix(runningBiasGrad[d], biasGrad[d]);
        }
    }
    @Override
    public void updateParameters(float learningRate, int trainingCount){
        updateWeights(learningRate, trainingCount);
        updateBiases(learningRate, trainingCount);
    }
    /** Subtracts from each weight, the average computed gradient for that weight times
     * the learning rate. The updated weights will be in the weights matrix of that layer. */
    private void updateWeights(float learningRate, int trainingCount) {
        for (int d = 0; d < weights.length; d++){
            for (int r = 1; r <= weights[0].getRows(); r++){
                for (int c = 1; c <= weights[0].getCols(); c++){
                    float dj_dw = runningWeightGrad[d].getElement(r, c)/trainingCount;
                    float weightUpdate = weights[d].getElement(r, c) - learningRate*dj_dw;
                    weights[d].setElement(r, c, weightUpdate);
                    // After the running grad is used, we set it to 0 to be used on another batch.
                    runningWeightGrad[d].setElement(r, c, 0f);
                }
            }
        }
    }
    private void updateBiases(float learningRate, int trainingCount) {
        for (int d = 0; d < biases.length; d++){
            for (int r = 1; r <= biases[0].getRows(); r++){
                for (int c = 1; c <= biases[0].getCols(); c++){
                    float dj_dw = runningBiasGrad[d].getElement(r, c)/trainingCount;
                    float biasUpdate = biases[d].getElement(r, c) - learningRate*dj_dw;
                    biases[d].setElement(r, c, biasUpdate);
                    // After the running grad is used, we set it to 0 to be used on another batch.
                    runningBiasGrad[d].setElement(r, c, 0f);
                }
            }
        }
    }
    /**
     * Returns a string representation of the layer.
     *
     * Output:
     * First Line: Conv3D rowFilterSize colFilterSize depthFilterSize rowStride colStride depthStride ActFunc flatten
     * Second Line: "list of weight matrices"
     * Third line: "list of bias matrices"
     *
     * So every new matrix on the same line would mean another element to the matrix list or 3-D.
     */
    @Override
    public List<String> stringLayer() {
        List<String> layerRep = new ArrayList<>();
        layerRep.add("Conv3D<>" + rowFilterSize + "<>" + colFilterSize + "<>" + depthFilterSize +
                "<>" + rowStride + "<>" + colStride + "<>" + depthStride + "<>" +
                act.name() + "<>" + flatten);
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

    @Override
    public void setWeights(Matrix[] weight) {
        weights = weight;
    }

    @Override
    public void setBiases(Matrix[] bias) {
        biases = bias;
    }

}
