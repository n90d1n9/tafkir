package tech.kayys.tafkir.training.strategy;

public class FineTuningConfig {
    private String methodology;
    private int loraRank;
    private float loraAlpha;
    private float dpoBeta;

    public FineTuningConfig(String methodology) {
        this.methodology = methodology;
    }

    public String getMethodology() {
        return methodology;
    }

    public int getLoraRank() {
        return loraRank;
    }

    public void setLoraRank(int loraRank) {
        this.loraRank = loraRank;
    }

    public float getLoraAlpha() {
        return loraAlpha;
    }

    public void setLoraAlpha(float loraAlpha) {
        this.loraAlpha = loraAlpha;
    }

    public float getDpoBeta() {
        return dpoBeta;
    }

    public void setDpoBeta(float dpoBeta) {
        this.dpoBeta = dpoBeta;
    }
}
