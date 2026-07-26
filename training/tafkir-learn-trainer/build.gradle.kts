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

sourceSets {
    named("main") {
        java.exclude("tech/kayys/aljabr/ml/runner/**")
    }
}

dependencies {
    api(project(":ml:tafkir-ml-runner-api"))
  //  api(project(":trainer:aljabr-trainer-api"))
  //  implementation(project(":trainer:aljabr-trainer"))
    api(project(":ml:tafkir-ml-autograd"))
    api(project(":ml:tafkir-ml-core"))
    api(project(":ml:tafkir-ml-data"))
    api(project(":ml:tafkir-ml-diffusion-api"))
    api(project(":ml:tafkir-ml-diffusion-opd"))
    api(project(":ml:tafkir-ml-nn"))
    api(project(":ml:tafkir-ml-estimator"))
    api(project(":ml:tafkir-ml-preprocessing"))
    api(project(":ml:tafkir-ml-selection"))
    api(project(":ml:tafkir-ml-optimize"))
    api(project(":ml:tafkir-ml-hub"))
    api(project(":ml:tafkir-ml-export"))
   // implementation(project(":sdk:aljabr-sdk-api"))
    api(project(":ml:tafkir-ml-multimodal"))
    api(project(":ml:tafkir-ml-cnn"))
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.assertj", name = "assertj-core", version = "3.26.3")
    testRuntimeOnly(group = "org.junit.platform", name = "junit-platform-launcher", version = "1.10.2")
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

tasks.jar {
    manifest {
        attributes(
            mapOf(
                "Automatic-Module-Name" to "tech.kayys.tafkir.ml"
            )
        )
    }
}

tasks.named<JavaCompile>("compileJava") {
    // The estimator project is mounted under an aliased physical directory.
    // Make the clean-build jar edge explicit so Gradle never races this compile
    // against the jar materialization required by the compile classpath.
    dependsOn(":ml:tafkir-ml-estimator:jar")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
