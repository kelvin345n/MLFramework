package Ep2.FrameworkML;

import java.util.Random;

/** Class that represents a matrix of rows x cols. Using a
 * single array of type T denoted by m.
 *
 * IMPORTANT NOTE: matrix is represented in row-major order
 *
 * */
public class Matrix {
    // Rows represents number of rows in matrix
    private int rows;
    // cols represents the number of cols in the matrix
    private int cols;
    // m is an array of type T that represents the elements in the matrix
    private float[] m;

    /** Constructor */
    public Matrix(int rows, int cols, float[] matrix){
        // Cannot create a matrix if the array length does not match
        // the parameters.
        if (matrix.length != cols * rows && cols * rows > 0) {
            throw new IllegalArgumentException("Invalid matrix dimensions: " +
                    "expected " + (cols * rows) + " elements, but got " + matrix.length);
        }
        this.cols = cols;
        this.rows = rows;
        this.m = matrix;

    }
    /** Creates a new matrix where all elements in the matrix are set to zero */
    public Matrix(int rows, int cols){
        this.cols = cols;
        this.rows = rows;
        this.m = new float[cols*rows];

        for (int r = 1; r <= rows; r++){
            for (int c = 1; c <= cols; c++){
                setElement(r, c, 0f);
            }
        }
    }

    public int getRows(){
        return rows;
    }
    public int getCols(){
        return cols;
    }
    public float[] getMatrixArray(){
        return m;
    }

    /** Returns a random float from 0 - N */
    public float getRandFloat(int low, int high){
        Random r = new Random();
        return r.nextFloat(low, high);
    }

    /** Randomizes all elements in the matrix based on a
     * seed value */
    public void randomizeMatrix(int low, int high) {
        for (int r = 1; r <= rows; r++) {
            for (int c = 1; c <= cols; c++) {
                setElement(r, c, getRandFloat(low, high));
            }
        }
    }

    /** Sets VALUE at the specified row and column in the matrix */
    public void setElement(int row, int col, float value){
        m[getIndex(row, col)] = value;
    }

    /** Given a row and column in the matrix, return an index into
     * the matrix array */
    private int getIndex(int row, int col){
        if (row < 1 || col < 1){
            throw new IllegalArgumentException("Invalid indices. Cannot be zero or negative");
        }
        if (row > rows || col > cols){
            throw new IllegalArgumentException("Invalid matrix dimensions: " +
                    "max dimension " + "(" + rows + ", " + cols + ")" + ", but tried to access "
                    + "(" + row + ", " + col + ")");
        }
        return ((row - 1) * cols + (col - 1));
    }

    /** Returns the element at the specified column and row of
     * the matrix. The matrix is one indexed. Meaning that col 1, row 1
     * is the first element of the array */
    public float getElement(int row, int col){
        return m[getIndex(row, col)];
    }

    /** Returns a sub-matrix that has all the elements at a certain row.
     * It will have the same num of rows as the parent matrix and 1 column*/
    public Matrix getColAt(int startCol){
        if (startCol <= 0 || startCol > cols){
            throw new IllegalArgumentException("Invalid column");
        }
        float[] matrixArray = new float[rows];
        for (int r = 1; r <= rows; r++){
            matrixArray[r-1] = getElement(r, startCol);
        }
        return new Matrix(rows, 1, matrixArray);
    }

    /** Returns a sub-matrix that gets all the elements at a certain row.
     * It will have 1 row and the columns of the parent matrix. */
    public Matrix getRowAt(int startRow){
        if (startRow <= 0 || startRow > rows){
            throw new IllegalArgumentException("Invalid row");
        }

        float[] matrixArray = new float[cols];
        for (int c = 1; c <= cols; c++){
            matrixArray[c-1] = getElement(startRow, c);
        }
        return new Matrix(1, cols, matrixArray);
    }

}
