plugins {
    java
}

dependencies {
    //implementation(project(":runner:safetensor:aljabr-safetensor-loader"))
    implementation(project(":quantizer:tafkir-quantizer-gptq"))
   // implementation(project(":runner:gguf:aljabr-gguf-core"))
    implementation("tech.kayys.aljabr:aljabr-tensor:0.1.0-SNAPSHOT")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("org.slf4j:slf4j-api:2.0.12")
}
