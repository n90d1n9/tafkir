plugins {
    java
    `maven-publish`
}

group = "tech.kayys.tafkir"
version = "0.3.0-SNAPSHOT"

dependencies {
    implementation(project(":core:aljabr-tensor"))      // Tensor, Shape, DType, DeviceType
    implementation(project(":core:aljabr-core"))         // Buffer, ManagedArena
    implementation(project(":backend:cpu:aljabr-backend-cpu")) // CpuBackend, CpuOps
    implementation(project(":autograd"))                 // AutogradEngine, GradRegistry, GGraph
    implementation(project(":core:aljabr-nn"))           // Linear, Sequential, ReLU (Aljabr's versions)
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

tasks.withType<JavaCompile> {
    options.compilerArgs.add("--enable-preview")
    options.compilerArgs.add("--add-modules=jdk.incubator.vector")
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            groupId = "tech.kayys.tafkir"
            artifactId = "tafkir-ml-aljabr"
            version = "0.3.0-SNAPSHOT"
        }
    }
}
