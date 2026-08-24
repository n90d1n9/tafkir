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
    implementation(project(":ml:tafkir-ml-nn"))
    implementation(project(":ml:tafkir-ml-autograd"))
    implementation(project(":core:alkhawarizm-tokenizer-core"))
   // implementation(project(":sdk:alkhawarizm-sdk-api"))
    implementation(project(":ml:tafkir-ml-optimize"))
    implementation(project(":ml:tafkir-ml-selection"))
    implementation(project(":ml:tafkir-ml-data"))
   // implementation(project(":spi:alkhawarizm-spi-inference"))
   // implementation(project(":spi:alkhawarizm-spi-multimodal"))
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
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
                "Automatic-Module-Name" to "tech.kayys.tafkir.ml.nlp"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
