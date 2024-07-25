package Ep2;

public class Launcher {

    public static void main(String[] strings){

        int count = 0;
        for (int i = 0; i < 100; i++){
            Xor b = new Xor();
            b.training();
            if (b.cost() > 0.05f){
                count++;
            }
        }
        System.out.println("Acc: " + count);

    }
}
