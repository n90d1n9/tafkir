plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.alkhawarizm"
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
    implementation(project(":core:alkhawarizm-core"))
    implementation(project(":core:alkhawarizm-tokenizer-core"))
    implementation(project(":ml:tafkir-ml-api"))
    implementation(project(":ml:tafkir-ml-autograd"))
    implementation(project(":ml:tafkir-ml-diffusion-opd"))
    implementation(project(":ml:tafkir-ml-nn"))
    implementation(project(":ml:tafkir-ml-cnn"))
    implementation(project(":ml:tafkir-ml-optimize"))
    implementation(project(":backend:metal:alkhawarizm-backend-metal"))
    implementation(project(":runner:alkhawarizm-diffusion"))
    implementation(project(":runner:safetensor:alkhawarizm-safetensor-loader"))
    implementation(project(":runner:safetensor:alkhawarizm-safetensor-quantization"))
    implementation(project(":runner:safetensor:alkhawarizm-runner-stable-diffusion"))
    implementation(project(":runner:safetensor:alkhawarizm-safetensor-core"))
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
