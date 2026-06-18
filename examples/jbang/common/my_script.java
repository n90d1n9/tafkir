//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;
import tech.kayys.tafkir.ml.autograd.GradTensor;

public class my_script {
    public static void main(String[] args) {
        System.out.println("=== Tafkir Neural Network ===");
        
        // TODO: Add your code here
        
        Sequential model = new Sequential(
            new Linear(784, 128),
            new ReLU(),
            new Linear(128, 10)
        );
        
        System.out.println("Model ready for training!");
    }
}
