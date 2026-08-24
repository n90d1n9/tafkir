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

dependencies {
    // Statistical models are pure Java — no tensor dependency required.
    // Neural models depend on tafkir-ml-core only at compile time (optional acceleration).
    compileOnly(project(":ml:tafkir-ml-core"))

    // Logging
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")

    // ── Tests ─────────────────────────────────────────────────────────────────
    testImplementation(platform("org.junit:junit-bom:5.11.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
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
