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
    implementation(project(":ml:tafkir-ml-nn"))
    //implementation(project(":core:alkhawarizm-runtime-config"))
    implementation(project(":core:alkhawarizm-model-repository"))
    implementation(project(":core:alkhawarizm-model-repo-hf"))
    implementation(project(":core:alkhawarizm-model-repo-local"))
   // implementation(project(":runner:safetensor:alkhawarizm-safetensor-loader"))
    implementation(project(":core:alkhawarizm-spi-model"))
    implementation("io.smallrye.reactive:mutiny")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.16.1")
    implementation("org.slf4j:slf4j-api:2.0.13")
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
                "Automatic-Module-Name" to "tech.kayys.tafkir.ml.hub"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
