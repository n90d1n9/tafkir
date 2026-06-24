rootProject.name = "tafkir-engine"

includeBuild("../aljabr")

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

include("quantizer:tafkir-quantizer-autoround")
include("quantizer:tafkir-quantizer-awq")
include("quantizer:tafkir-quantizer-gptq")
include("quantizer:tafkir-quantizer-quip")
include("quantizer:tafkir-quantizer-turboquant")

if (file("sdk/tafkir-sdk-session").isDirectory) {
//    include("sdk:tafkir-sdk-session")
}

include("ml:tafkir-ml-core")
include("ml:tafkir-ml-autograd")


includeOptionalProject("training:tafkir-train-strategy", "training/tafkir-train-strategy")
include("integration:jupyter:tafkir-jupyter-kernel")
