package Ep2.FrameworkML.ActivationFunctions;

import Ep2.FrameworkML.Matrix;

public class Tanh implements Activation{
    public float apply(float x, Matrix[] zValues){
        return 0f;
    }

    @Override
    public float derivative(float x, Matrix[] zValues) {
        return 0;
    }
    @Override
    public String name(){
        return "tanh";
    }
}
