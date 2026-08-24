plugins {
    java
}

dependencies {
    implementation("tech.kayys.alkhawarizm:alkhawarizm-core:0.1.0-SNAPSHOT")
}

dependencies {
    testImplementation("tech.kayys.alkhawarizm:alkhawarizm-rocksdb:0.1.0-SNAPSHOT")
    testImplementation("tech.kayys.alkhawarizm:alkhawarizm-helixdb:0.1.0-SNAPSHOT")
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
}
