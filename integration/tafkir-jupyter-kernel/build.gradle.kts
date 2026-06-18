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
    implementation(group = "org.dflib.jjava", name = "jjava", version = "1.0-a4")
    implementation(group = "tech.kayys.tafkir", name = "tafkir-ml-tensor")
    implementation(project(":ml:tafkir-ml-nn"))
    implementation(group = "tech.kayys.tafkir", name = "tafkir-ml-autograd")
    implementation(project(":ml:tafkir-ml-cnn"))
    implementation(project(":ml:tafkir-ml-data"))
    implementation(project(":ml:tafkir-ml-nlp"))
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
