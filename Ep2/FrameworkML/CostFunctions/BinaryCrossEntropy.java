package Ep2.FrameworkML.CostFunctions;

public class BinaryCrossEntropy implements Cost{
    @Override
    public float cost(float output, float expected) {
        float epsilon = 1e-5f; // Small constant to prevent log(0)
            // Clip predictions to the range [epsilon, 1 - epsilon]
        float clippedPred = Math.max(epsilon, Math.min(output, 1 - epsilon));
            // Compute the binary cross-entropy component for this sample
        float loss = expected * (float) Math.log10(clippedPred) + (1 - expected) * (float) Math.log10(1 - clippedPred);
        return -loss;
    }

    @Override
    public float derivative(float output, float expected) {
        float epsilon = 1e-5f; // Small constant to prevent division by zero
        // Clip predictions to the range [epsilon, 1 - epsilon]
        float clippedPred = Math.max(epsilon, Math.min(output, 1 - epsilon));
        // Compute the derivative of the binary cross-entropy loss
        float derivative = -(expected/clippedPred) + (1-expected)/(1-clippedPred);
        return derivative;
    }

    @Override
    public String name() {
        return "bce";
    }
}
