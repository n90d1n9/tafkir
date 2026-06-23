rootProject.name = "tafkir-engine"

// Aljabr is the compute engine — required composite build
includeBuild("../aljabr") {
    dependencySubstitution {
        substitute(module("tech.kayys.aljabr:aljabr-core"))
            .using(project(":core:aljabr-core"))
        substitute(module("tech.kayys.aljabr:aljabr-tensor"))
            .using(project(":core:aljabr-tensor"))
        substitute(module("tech.kayys.aljabr:aljabr-backend-cpu"))
            .using(project(":backend:cpu:aljabr-backend-cpu"))
        substitute(module("tech.kayys.aljabr:aljabr-nn"))
            .using(project(":core:aljabr-nn"))
        substitute(module("tech.kayys.aljabr:aljabr-autograd"))
            .using(project(":autograd"))
    }
}

fun includeOptionalProject(projectPath: String, vararg candidatePaths: String) {
    val projectDir = candidatePaths
        .map { file(it) }
        .firstOrNull { candidate ->
            candidate.resolve("build.gradle.kts").isFile || candidate.resolve("build.gradle").isFile
        }
        ?: return

    include(projectPath)
    project(":$projectPath").projectDir = projectDir
}

val staticallyIncludedModelProjects = setOf<String>()

includeOptionalProject("suling", "../extensions/audio/suling", "stubs/suling")

// New Aljabr-backed ML module (replaces old tafkir-ml-autograd)
include("ml:tafkir-ml-aljabr")

// Keep old autograd module temporarily for migration
include("ml:tafkir-ml-autograd")

// New trainer module with real training loop
include("trainer:tafkir-trainer-aljabr")
include("trainer:tafkir-trainer-api")

includeOptionalProject("training:tafkir-train-strategy", "training/tafkir-train-strategy")
