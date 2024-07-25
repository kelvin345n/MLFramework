package Ep2.FrameworkML.ActivationFunctions;

import Ep2.FrameworkML.Matrix;

public class Relu implements Activation{

    @Override
    public float apply(float x, Matrix[] zValues){
        return Math.max(0, x);
    }

    @Override
    public float derivative(float x, Matrix[] zValues) {
        if (x >= 0) return 1;
        return 0;
    }

    @Override
    public String name(){
        return "relu";
    }

}
