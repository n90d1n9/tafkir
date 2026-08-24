rootProject.name = "tafkir-engine"

// Aljabr is the compute engine — required composite build
includeBuild("../alkhawarizm") {
    dependencySubstitution {
        substitute(module("tech.kayys.alkhawarizm:alkhawarizm-core"))
            .using(project(":core:alkhawarizm-core"))
        substitute(module("tech.kayys.alkhawarizm:alkhawarizm-tensor"))
            .using(project(":core:alkhawarizm-tensor"))
        substitute(module("tech.kayys.alkhawarizm:alkhawarizm-backend-cpu"))
            .using(project(":backend:cpu:alkhawarizm-backend-cpu"))
        substitute(module("tech.kayys.alkhawarizm:alkhawarizm-nn"))
            .using(project(":core:alkhawarizm-nn"))
        substitute(module("tech.kayys.alkhawarizm:alkhawarizm-autograd"))
            .using(project(":autograd"))
    }
}

// Tafkir modules
include("ml:tafkir-ml-alkhawarizm")
include("trainer:tafkir-trainer-api")
include("trainer:tafkir-trainer-alkhawarizm")
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
