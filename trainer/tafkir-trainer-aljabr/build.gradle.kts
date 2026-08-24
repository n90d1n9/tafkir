plugins {
    java
    `maven-publish`
}

group = "tech.kayys.tafkir"
version = "0.3.0-SNAPSHOT"

dependencies {
    implementation(project(":ml:tafkir-ml-alkhawarizm"))
    implementation(project(":trainer:tafkir-trainer-api"))
    implementation(project(":core:alkhawarizm-tensor"))
    implementation(project(":core:alkhawarizm-nn"))
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
            artifactId = "tafkir-trainer-alkhawarizm"
            version = "0.3.0-SNAPSHOT"
        }
    }
}
