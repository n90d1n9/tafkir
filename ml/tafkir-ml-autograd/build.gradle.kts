plugins {
    `java-library`
}

dependencies {
    api("tech.kayys.aljabr:aljabr-tensor:0.1.0-SNAPSHOT")
    api(project(":ml:tafkir-ml-core"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("org.slf4j:slf4j-api:2.0.12")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter", version = "5.10.2")
    testRuntimeOnly(group = "org.junit.platform", name = "junit-platform-launcher", version = "1.10.2")
}

tasks.jar {
    manifest {
        attributes(
            mapOf(
                "Automatic-Module-Name" to "tech.kayys.tafkir.ml.autograd"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
