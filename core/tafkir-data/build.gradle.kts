plugins {
    java
    `maven-publish`
}

group = "tech.kayys.tafkir"
version = "0.3.0-SNAPSHOT"

dependencies {
    implementation(project(":ml:tafkir-ml-aljabr"))
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            groupId = "tech.kayys.tafkir"
            artifactId = "tafkir-data"
            version = "0.3.0-SNAPSHOT"
        }
    }
}
