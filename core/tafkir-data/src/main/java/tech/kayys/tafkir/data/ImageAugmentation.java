package tech.kayys.tafkir.data;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.util.Random;

/**
 * Simple data augmentation for image data.
 */
public final class ImageAugmentation {

    private final Random rng;

    public ImageAugmentation(long seed) { this.rng = new Random(seed); }
    public ImageAugmentation() { this.rng = new Random(); }

    public TafkirTensor randomHorizontalFlip(TafkirTensor images, float p) {
        if (rng.nextFloat() >= p) return images;
        float[] data = images.data();
        int batchSize = (int) images.shape()[0];
        float[] flipped = new float[data.length];
        for (int b = 0; b < batchSize; b++) {
            for (int r = 0; r < 28; r++) {
                for (int c = 0; c < 28; c++) {
                    int srcIdx = b * 784 + r * 28 + c;
                    int dstIdx = b * 784 + r * 28 + (27 - c);
                    flipped[dstIdx] = data[srcIdx];
                }
            }
        }
        return TafkirTensor.of(flipped, batchSize, 784);
    }

    public TafkirTensor addNoise(TafkirTensor images, float std) {
        float[] data = images.data();
        float[] noisy = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            noisy[i] = data[i] + (float) (rng.nextGaussian() * std);
            noisy[i] = Math.max(0, Math.min(1, noisy[i]));
        }
        return TafkirTensor.of(noisy, images.shapeArray());
    }
}
