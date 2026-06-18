//DEPS tech.kayys.tafkir:tafkir-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.nn.*;

public class error_handling {
    public static void main(String[] args) {
        try {
            Sequential model = new Sequential(
                new Linear(784, 128),
                new ReLU(),
                new Linear(128, 10)
            );
            
            System.out.println("Model created successfully!");
            
        } catch (IllegalArgumentException e) {
            System.err.println("Invalid model configuration: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
