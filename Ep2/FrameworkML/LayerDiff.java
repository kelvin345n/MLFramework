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
            return denseDiff(a, (Dense) next);
        } else if (next instanceof Conv3D) {
            return conv3dDiff(a, (Conv3D) next);
        } else {
            throw new IllegalArgumentException("Unsupported layer type: " + next.getClass().getName());
        }

    }
    /** Call this when the next layer is a dense layer. */
    private static Matrix[] denseDiff(Matrix[] a, Layer next){
        // Okay the next layer is a dense layer. That means each activation will be
        // connected to each of the dj/dz_next values.

        // We need to figure to first make sure that "a" has the same shape as
        // expected input.


        int[] aShape = new int[]{a[0].getRows(), a[0].getCols(), a.length};
        // If not equal then that means that activation value must have been reshaped
        // to interact with this layer.
        if (!Arrays.equals(aShape, next.getInputShape())){
            // We reassign the lcao
            a = Operations.reshape(a, next.getInputShape());
        }



        return a;
    }

    /** Call this when the next layer is a Conv3D layer */
    private static Matrix[] conv3dDiff(Matrix[] a, Layer next){

    }

}
