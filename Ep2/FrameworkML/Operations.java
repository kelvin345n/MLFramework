package Ep2.FrameworkML;

import Ep2.FrameworkML.ActivationFunctions.Activation;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;

/** Performs matrix operations */
public class Operations {

    /** Outputs a new matrix that is the dot product of matrix a
     * and matrix b. or AxB */
    public static Matrix dotMatrix(Matrix a, Matrix b){
        if (a.getCols() != b.getRows()){
            throw new IllegalArgumentException("Invalid matrix dimensions");
        }
        Matrix result = new Matrix(a.getRows(), b.getCols());
        for (int r = 1; r <= a.getRows(); r++){
            Matrix aSub = a.getRowAt(r);
            float[] aRows = aSub.getMatrixArray();
            for (int bc = 1; bc <= b.getCols(); bc++){
                Matrix bSub = b.getColAt(bc);
                float[] bCols = bSub.getMatrixArray();
                float product = 0;
                for (int i = 0; i < a.getCols(); i++){
                    product += aRows[i] * bCols[i];
                }
                result.setElement(r, bc, product);
            }
        }
        return result;
    }

    /** Adds matrix b to matrix a. matrix a will be changed while b is not
     * changed */
    public static void sumMatrix(Matrix a, Matrix b){
        if (a.getRows() != b.getRows() || a.getCols() != b.getCols()){
            throw new IllegalArgumentException("Matrices have incompatible dimensions");
        }

        for (int r = 1; r <= a.getRows(); r++){
            for (int c = 1; c <= a.getCols(); c++){
                float sum = a.getElement(r, c) + b.getElement(r, c);
                a.setElement(r, c, sum);
            }
        }
    }

    /** Prints all the contents of matrix a into the terminal */
    public static void printMatrix(Matrix a){
        for(int r = 1; r <= a.getRows(); r++){
            System.out.print("| ");
            for (int c = 1; c <= a.getCols(); c++){
                System.out.print(a.getElement(r, c) + " | ");
            }
            System.out.println();
        }

    }
    /** Returns a string representation of the given matrix */
    public static String stringMatrix(Matrix a){
        // First two numbers are the row and cols
        StringBuilder sb = new StringBuilder();
        // The rows, cols, and array string will be split using the ">" symbol.
        sb.append(a.getRows() + "<>" + a.getCols() + "<>" + Arrays.toString(a.getMatrixArray()));
        return sb.toString();
    }

    /** Applies and sets the sigmoud function to all elements of the given matrix */
    public static void sigmoudMatrix(Matrix a){
        for (int r = 1; r <= a.getRows(); r++){
            for (int c = 1; c <= a.getCols(); c++){
                float elem = a.getElement(r, c);
                a.setElement(r, c, sigmoudf(elem));
            }
        }
    }

    /** Takes the value of x and outputs a floating point value between
     * 0f and 1f. The smaller a number is, the closer it is to 0f. Vice-Versa */
    public static float sigmoudf(float x){
        return (float) (1.f / (1.f + Math.exp(-x)));
    }


    /** Creates a sub matrix from the given matrix that is size ROWS x COLS
     * and starts at the (rowStart, colStart) of the given matrix. Keep in mind
     * that the matrix is 1 indexed not 0 indexed */
    public static Matrix createSubMatrix(Matrix m, int rowStart, int colStart, int rowsSize, int colsSize){
        if (rowStart + rowsSize - 1 > m.getRows() || colStart + colsSize - 1 > m.getCols() || rowsSize == 0 || colsSize == 0){
            throw new IllegalArgumentException("Sub matrix invalid size or out of bounds error");
        }
        int count = 0;
        float[] subarray = new float[rowsSize * colsSize];
        for (int i = rowStart; i < rowStart + rowsSize; i++){
            for (int j = colStart; j < colStart + colsSize; j++){
                subarray[count] = m.getElement(i, j);
                count++;
            }
        }
        return new Matrix(rowsSize, colsSize, subarray);
    }

    /** Creates a sub tensor given a Tensor, where to start and how large the sub tensor should be */
    public static Matrix[] createSubTensor(Matrix[] m, int rowStart, int colStart, int depthStart,
                                           int rowSize, int colSize, int depthSize){
        if (m.length == 0){
            throw new IllegalArgumentException("Error. Tensor size 0");
        }
        if (rowStart + rowSize - 1 > m[0].getRows() || colStart + colSize - 1 > m[0].getCols() || rowSize == 0 || colSize == 0){

            System.out.println("rowStart:" + rowStart + "   colStart:" + colStart + "   depthStart:"+ depthStart);
            System.out.println("rowSize:" + rowSize + "     colSize:" + colSize + "     depthSize:" + depthSize);

            for (Matrix a : m){
                printMatrix(a);
                System.out.println();
            }
            throw new IllegalArgumentException("Out of bounds error");
        }
        if (depthStart + depthSize > m.length){
            System.out.println("rowStart:" + rowStart + "   colStart:" + colStart + "   depthStart:"+ depthStart);
            System.out.println("rowSize:" + rowSize + "     colSize:" + colSize + "     depthSize:" + depthSize);

            for (Matrix a : m){
                printMatrix(a);
                System.out.println();
            }
            throw new IllegalArgumentException("Out of bounds error. Not enough depth");
        }
        Matrix[] sub = new Matrix[depthSize];
        for (int d = 0; d < depthSize; d++){
            sub[d] = createSubMatrix(m[depthStart+d], rowStart, colStart, rowSize, colSize);
        }
        return sub;
    }



    /** Applies an activation function to every element in the given matrix */
    public static void activationFunction(Matrix mat, Activation actFunc, Matrix[] zValues){
        for (int r = 1; r <= mat.getRows(); r++){
            for (int c = 1; c <= mat.getCols(); c++){
                float elem = mat.getElement(r, c);
                mat.setElement(r, c, actFunc.apply(elem, zValues));
            }
        }
    }

    /** Applies the hadamard product (element-wise multiplication) on matrix a from matrix b.
     * a and b must be the same sized matrices. */
    public static Matrix hadamard(Matrix a, Matrix b){
        if (a.getRows() != b.getRows() || a.getCols() != b.getCols()){
            throw new IllegalArgumentException("Matrices must be the same size. " +
                                        "a = " + a.getRows() + " x " + a.getCols() +
                                        ";b = " + b.getRows() + " x " + b.getCols());
        }

        Matrix hadamard = new Matrix(a.getRows(), a.getCols());
        for (int r = 1; r <= hadamard.getRows(); r++){
            for (int c = 1; c <= hadamard.getCols(); c++){
                float a_rc = a.getElement(r, c);
                float b_rc = b.getElement(r, c);
                hadamard.setElement(r, c, a_rc * b_rc);
            }
        }
        return hadamard;
    }

    /** Sums up all the elements in the given matrix and returns a scalar value. */
    public static float elementSum(Matrix a){
        float sum = 0;
        for (int i = 1; i <= a.getRows(); i++){
            for (int j = 1; j <= a.getCols(); j++){
                sum += a.getElement(i, j);
            }
        }
        return sum;
    }

    public static Matrix copy(Matrix a){
        return new Matrix(a.getRows(), a.getCols(), a.getMatrixArray().clone());
    }

    /** Flattens all the elements in 3-D tensor "a" into a 1-Row, X-col, 1-Depth tensor.
     * Flattens it from the first matrix in row major order, and then moves to the second
     * matrix*/
    public static Matrix[] flatten(Matrix[] a){
        if (a.length == 0){
            throw new IllegalArgumentException("Tensor must be populated with matrices");
        }
        int depths = a.length;
        int rows = a[0].getRows();
        int cols = a[0].getCols();
        Matrix[] output = new Matrix[]{
                new Matrix(1, rows * cols * depths)
        };

        int countCol = 1;
        for (int d = 0; d < depths; d++){
            for (int r = 1; r <= rows; r++){
                for (int c = 1; c <= cols; c++){
                    float a_drc = a[d].getElement(r, c);
                    output[0].setElement(1, countCol++, a_drc);
                }
            }
        }
        return output;
    }

    public static Matrix[] reshape(Matrix[] a, int[] desiredShape){
        if (desiredShape.length > 3){
            throw new IllegalArgumentException("Reshape for more than 3 dimensions not supported. " +
                    "Because I suck at coding.");
        }
        if (a.length == 0){
            throw new IllegalArgumentException("Tensor must be populated with matrices");
        }
        int aDepth = a.length;
        int aRows = a[0].getRows();
        int aCols = a[0].getCols();

        int desireDepth = desiredShape[2];
        int desireRows = desiredShape[0];
        int desireCols = desiredShape[1];

        if (aDepth * aRows * aCols != desireDepth * desireRows * desireCols){
            throw new IllegalArgumentException("This tensor cannot be reshaped to " +
                    desiredShape.toString());
        }
        // Add all elements of matrix "a" into one large array.
        float[] combinedArray = new float[0];
        for (int d = 0; d < aDepth; d++){
            combinedArray = ArrayUtils.addAll(combinedArray, a[d].getMatrixArray());
        }
        int runCount = 0;
        Matrix[] result = new Matrix[desireDepth];
        for (int d = 0; d < desireDepth; d++){
            result[d] = new Matrix(desireRows, desireCols);
            for (int r = 1; r <= desireRows; r++){
                for (int c = 1; c <= desireCols; c++){
                    result[d].setElement(r, c, combinedArray[runCount++]);
                }
            }
        }
        return result;
    }

    /** Adds the given subTensor to tensor in-place. Where the addition
     *  starts at rowStart, colStart, and depthStart in tensor */
    public static void addSubTensor(Matrix[] tensor, Matrix[] subTensor,
                                    int rowStart, int colStart, int depthStart){

        int subDepth = subTensor.length;
        int tensorDepth = tensor.length;
        if (tensorDepth * subDepth == 0){
            throw new IllegalArgumentException("Tensors cannot have depth 0");
        }
        int subRow = subTensor[0].getRows();
        int subCol = subTensor[0].getCols();
        int tensorRow = tensor[0].getRows();
        int tensorCol = tensor[0].getCols();
        checkAddSubTensor(rowStart, colStart, depthStart, tensorRow,
                tensorCol, tensorDepth, subRow, subCol, subDepth);

        for (int d = 0; d < subDepth; d++){
            for (int r = 1; r <= subRow; r++){
                for (int c = 1; c <= subCol; c++){
                    float tVal = tensor[depthStart+d].getElement(rowStart+r-1, colStart+c-1);
                    float sVal = subTensor[d].getElement(r, c);
                    tensor[depthStart+d].setElement(rowStart+r-1, colStart+c-1, tVal + sVal);
                }
            }
        }
    }
    /** Provides a check for the inputs for 'addSubTensor' function */
    private static void checkAddSubTensor(int rowStart, int colStart, int depthStart,
                                          int tensorRow, int tensorCol, int tensorDepth,
                                          int subRow, int subCol, int subDepth){
        if (rowStart <= 0 || rowStart > tensorRow){
            throw new IllegalArgumentException("Row start is out of bounds for the given tensor");
        }
        if (colStart <= 0 || colStart > tensorCol){
            throw new IllegalArgumentException("Column start is out of bounds for the given tensor");
        }
        if (depthStart < 0 || depthStart >= tensorDepth){
            throw new IllegalArgumentException("Depth start is out of bounds for the given tensor");
        }
        if (subRow + rowStart - 1 > tensorRow){
            throw new IllegalArgumentException("row out of bounds error. The subtensor cannot start at " +
                    "row " + rowStart);
        }
        if (subCol + colStart - 1 > tensorCol){
            throw new IllegalArgumentException("Column out of bounds error. The subtensor cannot start at " +
                    "col " + colStart);
        }
        if (subDepth + depthStart - 1 >= tensorDepth){
            throw new IllegalArgumentException("Depth out of bounds error. The subtensor cannot start at " +
                    "depth " + depthStart);
        }
    }

    /** Scales up every element in 'tensor' IN-PLACE by the given scalar value.
     * Basically multiplies every value in tensor by scalar. */
    public static void scale(Matrix[] tensor, float scalar){
        for (int d = 0; d < tensor.length; d++){
            for (int r = 1; r <= tensor[0].getRows(); r++){
                for (int c = 1; c <= tensor[0].getCols(); c++){
                    float v = tensor[d].getElement(r, c);
                    tensor[d].setElement(r, c, scalar * v);
                }
            }
        }
    }




    public static void main(String[] args) {
        float[] g = new float[50*100];
        for (int i = 0; i < 50*100; i++){
            g[i] = (float)i;
        }
        Matrix a = new Matrix(50, 100, g);
        System.out.println(Operations.stringMatrix(a));
    }
}
