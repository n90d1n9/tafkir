package tech.kayys.tafkir.cli;

import io.quarkus.runtime.annotations.RegisterForReflection;
import tech.kayys.tafkir.spi.model.ModelConfig;
import tech.kayys.tafkir.safetensor.loader.SafetensorHeader;
import tech.kayys.tafkir.safetensor.loader.SafetensorTensorInfo;

@RegisterForReflection(targets = {
    ModelConfig.class,
    ModelConfig.RopeScaling.class,
    SafetensorHeader.class,
    SafetensorTensorInfo.class
})
public class NativeRegistrations {
}
