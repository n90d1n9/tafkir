plugins {
    java
    `maven-publish`
}

group = "tech.kayys.tafkir"
version = "0.3.0-SNAPSHOT"

dependencies {
    implementation(project(":ml:tafkir-ml-aljabr"))
    implementation(project(":trainer:tafkir-trainer-api"))
    implementation(project(":core:aljabr-tensor"))
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
            artifactId = "tafkir-trainer-aljabr"
            version = "0.3.0-SNAPSHOT"
        }
    }
}
