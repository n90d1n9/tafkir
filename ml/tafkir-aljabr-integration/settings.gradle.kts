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

// Tafkir modules
include("ml:tafkir-ml-aljabr")
include("trainer:tafkir-trainer-api")
include("trainer:tafkir-trainer-aljabr")
include("data:tafkir-data")
include("distributed:tafkir-distributed")
include("checkpoint:tafkir-checkpoint")
include("tafkir-cli")
include("quantizer:tafkir-quantizer-autoround")
include("quantizer:tafkir-quantizer-awq")
include("quantizer:tafkir-quantizer-gptq")
include("compiler:tafkir-compiler")
include("integration:jupyter:tafkir-jupyter-kernel")
include("examples:jbang")

// Optional: audio extension
includeOptionalProject("suling", "../extensions/audio/suling", "stubs/suling")

fun includeOptionalProject(name: String, dir: String, fallbackDir: String) {
    val projectDir = file(dir)
    if (projectDir.exists()) {
        include(name)
        project(":$name").projectDir = projectDir
    } else {
        val fallback = file(fallbackDir)
        if (fallback.exists()) {
            include(name)
            project(":$name").projectDir = fallback
        }
    }
}
