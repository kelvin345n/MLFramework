package Ep2.FrameworkML.CostFunctions;

public class MeanSquareError implements Cost{

    @Override
    public float cost(float output, float expected) {
        return (float) (Math.pow(output - expected, 2)/2);
    }

    @Override
    public float derivative(float output, float expected) {
        return (output - expected);
    }

    @Override
    public String name(){
        return "mse";
    }
}
