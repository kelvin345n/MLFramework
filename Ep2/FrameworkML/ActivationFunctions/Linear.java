package Ep2.FrameworkML.ActivationFunctions;

import Ep2.FrameworkML.Matrix;

public class Linear implements Activation{
    /**
     * Takes in a float and returns a float according to that function
     *
     * @param x
     */
    @Override
    public float apply(float x, Matrix[] zValues) {
        return x;
    }

    @Override
    public float derivative(float x, Matrix[] zValues) {
        return 1f;
    }

    @Override
    public String name() {
        return "linear";
    }
}
