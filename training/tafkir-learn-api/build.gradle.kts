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
    // Core Tafkir Learn modules
    api(project(":training:tafkir-learn-nn"))
    api(project(":training:tafkir-learn-optim"))
    api(project(":training:tafkir-learn-data"))
    api(project(":training:tafkir-learn-trainer"))
    
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    testImplementation("org.assertj:assertj-core:3.26.3")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.10.2")
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
                "Automatic-Module-Name" to "tech.kayys.tafkir.learn.api"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
