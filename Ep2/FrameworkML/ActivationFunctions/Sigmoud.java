package Ep2.FrameworkML.ActivationFunctions;

import Ep2.FrameworkML.Matrix;

public class Sigmoud implements Activation{

    @Override
    public float apply(float x, Matrix[] zValues){
        return (float) (1.f / (1.f + Math.exp(-x)));
    }

    @Override
    public float derivative(float x, Matrix[] zValues) {
        double e_x = Math.exp(-x);
        return (float)(e_x/Math.pow((1 + e_x), 2));
    }
    @Override
    public String name(){
        return "sigmoud";
    }
}
