plugins {
    java
    `maven-publish`
    id("com.diffplug.spotless") version "7.0.3" apply false
}

allprojects {
    group = "tech.kayys.tafkir"
    version = "0.3.0-SNAPSHOT"

    repositories {
        mavenCentral()
        mavenLocal()
    }
}

subprojects {
    apply(plugin = "java")
    apply(plugin = "maven-publish")

    java {
        toolchain {
            languageVersion.set(JavaLanguageVersion.of(25))
        }
        withSourcesJar()
        withJavadocJar()
    }

    tasks.withType<JavaCompile> {
        options.compilerArgs.add("--enable-preview")
        options.compilerArgs.add("--add-modules=jdk.incubator.vector")
    }
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            pom {
                name.set("Tafkir Engine")
                description.set("Java ML/AI Framework built on Aljabr")
                url.set("https://github.com/n90d1n9/tafkir")
                licenses {
                    license {
                        name.set("Apache-2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                    }
                }
                scm {
                    connection.set("scm:git:git://github.com/n90d1n9/tafkir.git")
                    developerConnection.set("scm:git:ssh://github.com/n90d1n9/tafkir.git")
                    url.set("https://github.com/n90d1n9/tafkir")
                }
                developers {
                    developer {
                        id.set("n90d1n9")
                        name.set("Tafkir Team")
                    }
                }
            }
        }
    }
}
