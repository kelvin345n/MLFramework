package Ep2.FrameworkML.ActivationFunctions;

import Ep2.FrameworkML.Matrix;

public class Softmax implements Activation{
    /**
     * Takes in a float and returns a float according to that function
     *
     * @param x
     */
    @Override
    public float apply(float x, Matrix[] zValues) {
        float denom = 0f;
        for (int d = 0; d < zValues.length; d++){
            for (int r = 1; r <= zValues[d].getRows(); r++){
                for (int c = 1; c <= zValues[d].getCols(); c++){
                    denom += (float) Math.exp(zValues[d].getElement(r, c));
                }
            }
        }
        float num = (float) Math.exp(x);
        return num/denom;
    }

    @Override
    public float derivative(float x, Matrix[] zValues) {
        double ex = Math.exp(x);
        double sum = 0;
        for (int d = 0; d < zValues.length; d++){
            for (int r = 1; r <= zValues[d].getRows(); r++){
                for (int c = 1; c <= zValues[d].getCols(); c++){
                    float z_drc = zValues[d].getElement(r, c);
                    sum += Math.exp(z_drc);
                }
            }
        }
        double num = (sum - ex)*ex;
        double denom = Math.pow(sum, 2);
        return (float) (num/denom);
    }

    @Override
    public String name(){
        return "softmax";
    }
}
