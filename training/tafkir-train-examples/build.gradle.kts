plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    implementation(project(":core:aljabr-core"))
    implementation(project(":core:aljabr-tokenizer-core"))
    implementation(project(":ml:tafkir-ml-api"))
    implementation(project(":ml:tafkir-ml-autograd"))
    implementation(project(":ml:tafkir-ml-diffusion-opd"))
    implementation(project(":ml:tafkir-ml-nn"))
    implementation(project(":ml:tafkir-ml-cnn"))
    implementation(project(":ml:tafkir-ml-optimize"))
    implementation(project(":backend:metal:aljabr-backend-metal"))
    implementation(project(":runner:aljabr-diffusion"))
    implementation(project(":runner:safetensor:aljabr-safetensor-loader"))
    implementation(project(":runner:safetensor:aljabr-safetensor-quantization"))
    implementation(project(":runner:safetensor:aljabr-runner-stable-diffusion"))
    implementation(project(":runner:safetensor:aljabr-safetensor-core"))
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.assertj", name = "assertj-core")
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
        }
    }
    repositories {
        mavenLocal()
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
