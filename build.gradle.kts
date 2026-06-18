import org.gradle.api.tasks.SourceSetContainer
import org.gradle.api.publish.PublishingExtension
import org.gradle.api.publish.maven.MavenPublication

buildscript {
    repositories {
        google()
        mavenCentral()
        mavenLocal()
    }
    dependencies {
        classpath("io.smallrye:jandex:3.5.3")
    }
}

plugins {
    java
    `maven-publish`
    id("io.quarkus") version "3.32.2" apply false
}

extra["tafkirVersion"] = "0.1.0-SNAPSHOT"
extra["quarkusVersion"] = "3.32.2"

allprojects {
    group = "tech.kayys.tafkir"
    version = rootProject.extra["tafkirVersion"] as String

    repositories {
        google()
        mavenCentral()
        mavenLocal()
    }
}

subprojects {
    apply(plugin = "java")
    apply(plugin = "maven-publish")

    val quarkusVersion = rootProject.extra["quarkusVersion"] as String
    val mutinyVersion = "2.5.5"
    val smallryeMutinyVertxVersion = "3.15.1"
    val caffeineVersion = "3.1.8"
    val commonsCollectionsVersion = "4.4"
    val jakartaValidationVersion = "3.0.2"
    val jakartaEnterpriseVersion = "4.0.1"
    val jakartaInjectVersion = "2.0.1"
    val reactiveStreamsVersion = "1.0.4"
    val jbossLoggingVersion = "3.6.1.Final"
    val jacksonVersion = "2.16.1"
    val junitJupiterVersion = "5.10.2"
    val junitPlatformVersion = "1.10.2"
    val mockitoJupiterVersion = "5.14.2"

    java {
        toolchain {
            languageVersion.set(JavaLanguageVersion.of(25))
        }
    }

    dependencies {
        add("testRuntimeOnly", "org.junit.platform:junit-platform-launcher:$junitPlatformVersion")
        add("testImplementation", platform("org.junit:junit-bom:$junitJupiterVersion"))
        add("testImplementation", "org.junit.jupiter:junit-jupiter")
        add("testImplementation", "org.assertj:assertj-core:3.25.3")
    }

    tasks.withType<JavaCompile> {
        options.encoding = "UTF-8"
        options.compilerArgs.add("--enable-preview")
        options.compilerArgs.add("--add-modules=jdk.incubator.vector")
    }

    tasks.withType<Test>().configureEach {
        jvmArgs(
            "--enable-preview",
            "--add-modules=jdk.incubator.vector",
            "--enable-native-access=ALL-UNNAMED",
        )
        useJUnitPlatform()
    }

    tasks.withType<JavaExec>().configureEach {
        jvmArgs(
            "--enable-preview",
            "--add-modules=jdk.incubator.vector",
            "--enable-native-access=ALL-UNNAMED",
        )
    }

    val sourceSets = extensions.getByType(SourceSetContainer::class.java)
    val mainSourceSet = sourceSets.named("main")
    val jandexOutputDir = layout.buildDirectory.dir("generated/jandex/main")

    val generateJandexIndex = tasks.register("generateJandexIndex") {
        dependsOn(tasks.named("compileJava"), tasks.named("processResources"))

        val classesDirs = mainSourceSet.map { it.output.classesDirs }
        val indexFile = jandexOutputDir.map { it.file("META-INF/jandex.idx") }

        inputs.files(classesDirs)
        outputs.file(indexFile)

        doLast {
            val indexer = org.jboss.jandex.Indexer()

            classesDirs.get().files
                .filter { it.exists() }
                .forEach { classesDir ->
                    classesDir.resolve("META-INF/jandex.idx").let { staleIndex ->
                        if (staleIndex.exists()) {
                            staleIndex.delete()
                        }
                    }
                    classesDir.walkTopDown()
                        .filter { it.isFile && it.extension == "class" }
                        .forEach { classFile ->
                            classFile.inputStream().use(indexer::index)
                        }
                }

            val indexOutput = indexFile.get().asFile
            indexOutput.parentFile.mkdirs()
            indexOutput.outputStream().use { output ->
                org.jboss.jandex.IndexWriter(output).write(indexer.complete())
            }
        }
    }

    mainSourceSet.configure {
        output.dir(mapOf("builtBy" to generateJandexIndex), jandexOutputDir)
    }

    tasks.named("classes") {
        dependsOn(generateJandexIndex)
    }

    configurations.configureEach {
        resolutionStrategy.eachDependency {
            if (requested.group == "io.quarkus" && requested.version.isNullOrBlank()) {
                useVersion(quarkusVersion)
            }
            if (requested.group == "io.smallrye.reactive"
                && requested.name == "mutiny"
                && requested.version.isNullOrBlank()) {
                useVersion(mutinyVersion)
            }
            if (requested.group == "com.fasterxml.jackson.core"
                && (requested.name == "jackson-core" || requested.name == "jackson-databind")
                && requested.version.isNullOrBlank()) {
                useVersion(jacksonVersion)
            }
        }
    }

    afterEvaluate {
        extensions.configure<PublishingExtension>("publishing") {
            repositories {
                mavenLocal()
            }
            if (publications.findByName("mavenJava") == null) {
                publications.create<MavenPublication>("mavenJava") {
                    from(components["java"])
                }
            }
        }
    }
}
