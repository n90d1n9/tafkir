package tech.kayys.tafkir.training.strategy;

public class FineTuningFactory {

    public static FineTuningStrategy create(FineTuningConfig config) {
        String method = config.getMethodology() != null ? config.getMethodology().toLowerCase() : "";

        switch (method) {
            case "lora":
            case "qlora":
                return new LoRAStrategy(config.getLoraRank(), config.getLoraAlpha());
            case "dpo":
                return new DpoStrategy(config.getDpoBeta());
            case "full":
                return new FullParameterStrategy();
            default:
                throw new IllegalArgumentException("Unknown methodology: " + config.getMethodology());
        }
    }
}
