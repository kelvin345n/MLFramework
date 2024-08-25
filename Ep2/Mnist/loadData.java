package Ep2.Mnist;
import Ep2.FrameworkML.Matrix;
import net.sf.saxon.expr.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.io.*;

public class loadData {
    /** Takes in the file name for a csv file. Returns a list of inputs and their
     * corresponding pair. Where the first element is a list of inputs, and the
     * second element is the list of outputs. */
    public static List<List<Matrix[]>> load(String csv) {
        try {
            List<Matrix[]> inputs = new ArrayList<>();
            List<Matrix[]> outputs = new ArrayList<>();

            List<List<Matrix[]>> ret = new ArrayList<>();
            ret.add(inputs);
            ret.add(outputs);

            Scanner scan = new Scanner(new File(csv));
            if (scan.hasNextLine()){
                // Remove the titles
                scan.nextLine();
            }
            int count = 0;
            while (scan.hasNextLine()){
                count++;
                if (count % 5000 == 0){
                    System.out.println(count);
                }
                String line = scan.nextLine();
                String[] nums = line.split(",");
                // First element is the y value for the mnist dataset
                float[] yValues = new float[10];
                yValues[Integer.parseInt(nums[0])] = 1.0f;
                Matrix y = new Matrix(1, 10, yValues);
                // The rest of the elements are the x values
                float[] xValues = new float[nums.length-1];
                for (int i = 1; i < nums.length; i++){
                    xValues[i-1] = Float.parseFloat(nums[i]);
                }
                Matrix x = new Matrix(1, xValues.length, xValues);
                inputs.add(new Matrix[]{x});
                outputs.add(new Matrix[]{y});
            }
            scan.close();
            return ret;
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
}
