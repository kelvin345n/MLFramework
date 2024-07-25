package Ep2.FrameworkML;

import Ep2.FrameworkML.ActivationFunctions.Relu;
import Ep2.FrameworkML.Layers.Conv3D;
import Ep2.FrameworkML.Layers.Dense;
import Ep2.FrameworkML.Layers.Layer;

import java.util.Arrays;

/** This class will be used to compute the dj/da of an activation function depending on the layer
 * that is next. This is assuming that the "next" layer has had its differentiation completed.
 * Therefore, next.getDJ_DZ will get that layer's dj/dz tensor.
 * */
public class LayerDiff {

    /** Given a tensor of activation values that are inputs to the "next" layer,
     * this method returns a tensor of the same shape as "a" but with each of its values
     * replaced with dj/da */
    public static Matrix[] actDiff(Matrix[] a, Layer next){
        if (next instanceof Dense) {
            return denseActDiff(a, next);
        } else if (next instanceof Conv3D) {
            return conv3dActDiff(a, next);
        } else {
            throw new IllegalArgumentException("Unsupported layer type: " + next.getClass().getName());
        }

    }
    /** Call this when the next layer is a dense layer. Returns a tensor with dj/da values
     * for the given tensor.  */
    private static Matrix[] denseActDiff(Matrix[] a, Layer next){
        // Okay the next layer is a dense layer. That means each activation will be
        // connected to each of the dj/dz_next values.
        // We need to figure to first make sure that "a" has the same shape as
        // expected input.
        int[] aShape = new int[]{a[0].getRows(), a[0].getCols(), a.length};
        // If not equal then that means that activation value must have been reshaped
        // to interact with this layer.
        if (!Arrays.equals(aShape, next.getInputShape())){
            // We reassign the local a variable.
            a = Operations.reshape(a, next.getInputShape());
        }
        // Get the weights from the next layer because each activation in the current layer
        // was multiplied by the weights in the next layer.
        Matrix nextWeights = next.getWeights()[0];
        for (int c = 1; c <= a[0].getCols(); c++){
            // Getting all the weights that the activation at "c" is multiplied by.
            Matrix weightsForA_c = nextWeights.getRowAt(c);
            // We do hadamard product of each of these weights and the dj_dz matrix
            // of the next layer
            Matrix dj_dz_nextLayer = next.getDJ_DZ()[0];
            Matrix dj_dac = Operations.hadamard(weightsForA_c, dj_dz_nextLayer);
            a[0].setElement(1, c, Operations.elementSum(dj_dac));
        }
        // We must reshape it back if it was reshaped.
        if (!Arrays.equals(aShape, next.getInputShape())){
            a = Operations.reshape(a, aShape);
        }
        // Each of the elements in "a" should now have its derivative of j w/rt to itself.
        return a;
    }

    /** Call this when the next layer is a Conv3D layer. Returns a tensor with dj/da values
     * for the given tensor. */
    private static Matrix[] conv3dActDiff(Matrix[] a, Layer next){
        // Okay the next layer is a Conv3D layer. That means each activation will be
        // connected to only some of the dj/dz_next values. Depending on the filter
        // that was used on it.

        // We don't need to know any of the values inside the activation tensor,
        // so we set it to zero, so that we can add all the dj/dz * w, to it.
        int aRow = a[0].getRows();
        int aCol = a[0].getCols();
        int aDepth = a.length;
        for (int d = 0; d < aDepth; d++){
            a[d] = new Matrix(aRow, aCol);
        }
        // We need to figure to first make sure that "a" has the same shape as
        // expected input.
        int[] aShape = new int[]{aRow, aCol, aDepth};
        // If not equal then that means that activation tensor must have been reshaped
        // to interact with this layer.
        if (!Arrays.equals(aShape, next.getInputShape())){
            // We reassign the local a variable.
            a = Operations.reshape(a, next.getInputShape());
        }

        // Get the weights from the next layer because each activation in the current layer
        // was multiplied by the weights in the next layer.
        Matrix[] nextWeights = next.getWeights();
        // Also need the next layer's dj/dz to multiply with the weight.
        Matrix[] next_dj_dz = next.getDJ_DZ();
        // Okay we have the dj/dz and weights. Now we need to figure out what
        // activation values go with what dj/dz values and the weight that was used to
        // on that activation value.

        // What we are going to do is go through each filter, get its associated dj/dz value,
        // And multiply all values in that filter by that dj/dz value.

        // Okay so we need to figure out filter size.
        int n_weightDepth = nextWeights.length;
        int n_weightRows = nextWeights[0].getRows();
        int n_weightCols = nextWeights[0].getCols();
        int n_djdzDepth = next_dj_dz.length;
        int n_djdzRows = next_dj_dz[0].getRows();
        int n_djdzCols = next_dj_dz[0].getCols();
        int activDepth = a.length;
        int activRows = a[0].getRows();
        int activCols = a[0].getCols();
        // The total number of filters used on the activation tensor.
        int n_filtersUsed = n_djdzDepth * n_djdzRows * n_djdzCols;
        // The filter size that was used on our activation tensor.
        int n_filterDepthSize = n_weightDepth;
        int n_filterRowSize = n_weightRows;
        int n_filterColSize = n_weightCols/n_filtersUsed;
        // Okay we now know what size filter was used on the activation tensor.
        // Now we need to figure out the stride lengths that was used on the activation tensor.
        int rowStride = calculateStride(activRows, n_filterRowSize, n_djdzRows);
        int colStride = calculateStride(activCols, n_filterColSize, n_djdzCols);
        int depthStride = calculateStride(activDepth, n_filterDepthSize, n_djdzDepth);

        // Figure out how many filters are done on each dimension
        int numOfColFilters = (aCol - n_filterColSize)/colStride + 1;
        int numOfRowFilters = (aRow - n_filterRowSize)/rowStride + 1;
        int numOfDepthFilters = (aDepth - n_filterDepthSize)/depthStride + 1;

        // We checking
        if (numOfDepthFilters != n_djdzDepth){
            throw new IllegalArgumentException("Wrongger");
        }
        if (numOfRowFilters != n_djdzRows){
            throw new IllegalArgumentException("Wrongger");
        }
        if (numOfColFilters != n_djdzCols){
            throw new IllegalArgumentException("Wrongger");
        }

        // Start at the first filter. filter row start always stays 1.
        // Filter depth start always stays at 0.
        int filterRowStart = 1;
        int filterColStart = 1;
        int filterDepthStart = 0;
        // The output should be shape (numOfRowFilters, numOfColFilters, numOfDepthFilters)
        int inputDepthStart = 0;
        for (int d = 0; d < numOfDepthFilters; d++){
            // We need to apply filter starting at rowStart and colStart.
            int inputRowStart = 1;
            // Apply the filter to each sub tensor of the input.
            for (int r = 1; r <= numOfRowFilters; r++){
                int inputColStart = 1;
                for (int c = 1; c <= numOfColFilters; c++){
                    float associatedZ = next_dj_dz[d].getElement(r, c);
                    Matrix[] filter = Operations.createSubTensor(nextWeights,
                            filterRowStart, filterColStart, filterDepthStart,
                            n_filterRowSize, n_filterColSize, n_filterDepthSize
                    );
                    // We have the filter that was used at inputRowStart, inputcol, and input depth start.
                    // now we multiply dj/dz by every element in filter.
                    Operations.scale(filter, associatedZ);
                    Operations.addSubTensor(a, filter, inputRowStart, inputColStart, inputDepthStart);

                    // the filter size stays the same after each operation.
                    // Update the column start for both
                    filterColStart += n_filterColSize;
                    inputColStart += colStride;
                }
                // Keep in mind we do not need to check if we run out of bounds
                // with inputRowStart because we checked already how many row filters
                // are used in numOfRowFilters
                inputRowStart += rowStride;
            }
            inputDepthStart += depthStride;
        }
        // Each of the elements in "a" should now have its derivative of j w/rt to itself.
        // We reshape a back into its original shape.
        if (!Arrays.equals(aShape, next.getInputShape())){
            a = Operations.reshape(a, aShape);
        }
        return a;
    }

    /** Given the input size, filter size, and output size of that respective dimension,
     * we return the stride size that the filter takes with respect to that dimension */
    private static int calculateStride(int inputSize, int filterSize, int outputSize){
        // It just means that the filter spans the entire input size for that dimension
        // So we just return 1 for the stride length
        if (outputSize == 1 || inputSize == filterSize){
            return 1;
        }
        return (inputSize - filterSize)/(outputSize - 1);
    }

}
