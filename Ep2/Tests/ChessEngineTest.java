package Ep2.Tests;

import Ep2.FrameworkML.ActivationFunctions.Relu;
import Ep2.FrameworkML.ActivationFunctions.Softmax;
import Ep2.FrameworkML.CostFunctions.MeanSquareError;
import Ep2.FrameworkML.Layers.Conv3D;
import Ep2.FrameworkML.Layers.Dense;
import Ep2.FrameworkML.Layers.Layer;
import Ep2.FrameworkML.Matrix;
import Ep2.FrameworkML.NeuralNet2;
import Ep2.FrameworkML.NeuralNetReader;
import Ep2.FrameworkML.Operations;
import net.sf.saxon.expr.Component;
import org.junit.jupiter.api.Test;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
/** Pre-tests before actual training. To determine if the architecture
 * would actually learn. */
public class ChessEngineTest {
    /**
     *                              NEURAL NETWORK ARCHITECTURE
     *
     * INPUT LAYER:
     *  10 x 10 w/ 14 layers, w/ padding of size 1 to all sides.
     *
     *  Layer 1: One hot encoding of king of color next to act
     *  Layer 2: One hot encoding of pawn of color next to act
     *  Layer 3: One hot encoding of knight of color next to act
     *  Layer 4: One hot encoding of bishop of color next to act
     *  Layer 5: One hot encoding of rook of color next to act
     *  Layer 6: One hot encoding of queen of color next to act
     *  Layer 7: One hot encoding of all squares that the next to act color is attacking
     *
     *  Layer 8: One hot encoding of king of opponent
     *  Layer 9: One hot encoding of pawn of opponent
     *  Layer 10: One hot encoding of knight of opponent
     *  Layer 11: One hot encoding of bishop of opponent
     *  Layer 12: One hot encoding of rook of opponent
     *  Layer 13: One hot encoding of queen of opponent
     *  Layer 14: One hot encoding of all squares that the opponent is attacking
     *
     * HIDDEN LAYER 1 (Conv3D w/ Relu):
     * Filter: 3 x 3 x 7 filter w/ 1 row stride, 1 col stride, 7 depth stride.
     *          Thought process: This allows for this layer to only take in and evaluate the
     *                          board for each color. The next layer will determine analyze the board
     *                          as a whole.
     *
     * Output: 8 x 8 x 2
     *
     * HIDDEN LAYER 2 (Conv3D w/ Relu):
     * Filter: 3x3x2 filter w/ rowStride=1, colStride=1, depthStride=1
     *
     *      ThoughtProcess: This layer will analyze the position between the two colors.
     *
     * Output: 6 x 6 x 1 w/ flatten -----> 1 x 36 x 1
     *
     * HIDDEN LAYER 3 (Dense w/ Relu):
     * Neuron Count: 72
     *      ThoughtProcess: We need double the inputs to capture the complexity of the game.
     *
     * Output: 1 x 72 x 1
     *
     * HIDDEN LAYER 4 (Dense w/ Relu):
     * Neuron Count 36:
     *      ThoughtProcess: We now decipher the complexity back into its original size.
     *
     * HIDDEN LAYER 5 (Dense w/ softmax):
     * Neuron Count: 3
     *      ThoughtProcess: This vector will correspond to the probability of (win, draw, lose) for
     *                      the current player.
     * Output: 1 x 3 x 1
     *
     * */

    // Okay lets build baby

    @Test
    public void saveAndLoadArchitecture(){
        Layer[] arch = new Layer[]{
                new Conv3D(3, 3, 7, 1, 1, 7, new Relu()),
                new Conv3D(3, 3, 2, 1, 1, 1, new Relu(), true),
                new Dense(72, new Relu()),
                new Dense(36, new Relu()),
                new Dense(3, new Softmax())
        };
        int[] inputShape = new int[]{10, 10, 14};
        int[] outputShape = new int[]{1, 3, 1};

        NeuralNet2 engine = new NeuralNet2(inputShape, outputShape, arch, new MeanSquareError());
        StringBuilder original = new StringBuilder();
        for (String s: engine.getNetworkString()){
            System.out.println(s);
            original.append(s);
        }
        engine.saveNetwork("engineTest");
        NeuralNet2 loadEngine = NeuralNetReader.loadNetwork("engineTest");
        System.out.println("\nLOADED ENGINE");
        StringBuilder loaded = new StringBuilder();
        for (String s: loadEngine.getNetworkString()){
            System.out.println(s);
            loaded.append(s);
        }
        assertThat(loaded.toString()).isEqualTo(original.toString());

        NeuralNetReader.deleteSave("engineTest");
    }

    @Test
    public void architectureTest(){
        // Put in a position and
        Layer[] arch = new Layer[]{
                new Conv3D(3, 3, 7, 1, 1, 7, new Relu()),
                new Conv3D(3, 3, 2, 1, 1, 1, new Relu(), true),
                new Dense(72, new Relu()),
                new Dense(36, new Relu()),
                new Dense(3, new Softmax())
        };
        int[] inputShape = new int[]{10, 10, 14};
        int[] outputShape = new int[]{1, 3, 1};

        NeuralNet2 engine = new NeuralNet2(inputShape, outputShape, arch, new MeanSquareError());

        // Training Data:

        Matrix[] inputLayer = new Matrix[14];
                // Layer 1: One hot encoding of king of color next to act
                // Layer 2: One hot encoding of pawn of color next to act
                // Layer 3: One hot encoding of knight of color next to act
                // Layer 4: One hot encoding of bishop of color next to act
                // Layer 5: One hot encoding of rook of color next to act
                // Layer 6: One hot encoding of queen of color next to act
                // Layer 7: One hot encoding of all squares that the next to act color is attacking

                // Layer 8: One hot encoding of king of opponent
                // Layer 9: One hot encoding of pawn of opponent
                // Layer 10: One hot encoding of knight of opponent
                // Layer 11: One hot encoding of bishop of opponent
                // Layer 12: One hot encoding of rook of opponent
                // Layer 13: One hot encoding of queen of opponent
                // Layer 14: One hot encoding of all squares that the opponent is attacking








    }



}
