///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.tafkir:tafkir-ml-ml:0.3.0-SNAPSHOT
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/tafkir

import tech.kayys.tafkir.ml.Tafkir;

public class kernel_test {
    public static void main(String[] args) {
        System.out.println("=== Tafkir Kernel Test ===");
        System.out.println("Java version: " + System.getProperty("java.version"));
        Tafkir.printInfo();
        System.out.println("✓ Kernel working!");
    }
}
