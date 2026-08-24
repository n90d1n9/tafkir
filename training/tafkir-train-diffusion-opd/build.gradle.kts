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
    api(project(":ml:tafkir-ml-diffusion-api"))
   /*  implementation(project(":core:alkhawarizm-core"))
    implementation(project(":runner:alkhawarizm-diffusion"))
    implementation(project(":runner:safetensor:alkhawarizm-runner-stable-diffusion"))
    implementation(project(":runner:safetensor:alkhawarizm-safetensor-core")) */
    implementation(project(":backend:metal:alkhawarizm-backend-metal"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
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
                "Automatic-Module-Name" to "tech.kayys.tafkir.ml.diffusion.opd"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
