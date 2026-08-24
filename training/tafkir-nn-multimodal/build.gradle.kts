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
   // implementation(project(":sdk:alkhawarizm-sdk-api"))
    implementation(project(":ml:tafkir-ml-autograd"))
    implementation(project(":ml:tafkir-ml-nn"))
    implementation(project(":core:alkhawarizm-spi-model"))
  //  implementation(project(":spi:alkhawarizm-spi-multimodal"))
   // implementation(project(":spi:alkhawarizm-spi-inference"))
    implementation(group = "org.slf4j", name = "slf4j-api", version = "2.0.13")
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
                "Automatic-Module-Name" to "tech.kayys.alkhawarizm.lib.multimodal"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
