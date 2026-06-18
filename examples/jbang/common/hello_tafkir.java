//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;

public class hello_tafkir {
    public static void main(String[] args) {
        System.out.println("Hello from Tafkir!");
        
        // Create a simple model
        Sequential model = new Sequential(
            new Linear(10, 5),
            new ReLU(),
            new Linear(5, 1)
        );
        
        System.out.println("Model created successfully!");
    }
}
