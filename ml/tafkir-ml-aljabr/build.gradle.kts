plugins {
    java
    `maven-publish`
}

group = "tech.kayys.tafkir"
version = "0.3.0-SNAPSHOT"

dependencies {
    implementation(project(":core:aljabr-tensor"))
    implementation(project(":core:aljabr-core"))
    implementation(project(":backend:cpu:aljabr-backend-cpu"))
    implementation(project(":autograd"))
    implementation(project(":core:aljabr-nn"))
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
