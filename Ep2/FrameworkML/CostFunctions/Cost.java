package Ep2.FrameworkML.CostFunctions;

public interface Cost {

    float cost(float output, float expected);

    float derivative(float output, float expected);

    String name();
}
