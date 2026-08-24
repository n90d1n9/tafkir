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
    implementation(project(":ml:tafkir-ml-autograd"))
    implementation(project(":core:alkhawarizm-spi-model"))
    //implementation(project(":spi:alkhawarizm-spi-multimodal"))
    implementation(project(":core:alkhawarizm-tokenizer-core"))
    implementation("com.fasterxml.jackson.core:jackson-databind")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
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
                "Automatic-Module-Name" to "tech.kayys.tafkir.ml.data"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
