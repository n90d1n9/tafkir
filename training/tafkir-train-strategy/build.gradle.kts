plugins {
    java
}

dependencies {
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
}

dependencies {
    testImplementation(project(":aljabr:core:aljabr-rocksdb"))
    testImplementation(project(":aljabr:core:aljabr-helixdb"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
}
