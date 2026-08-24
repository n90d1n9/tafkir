plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.tafkir"
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
    api("tech.kayys.alkhawarizm:alkhawarizm-core:0.1.0-SNAPSHOT")
    api("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    api("tech.kayys.alkhawarizm:alkhawarizm-backend-cpu:0.1.0-SNAPSHOT")
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
                "Automatic-Module-Name" to "tech.kayys.tafkir.ml.core"
            )
        )
    }
}
