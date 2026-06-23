package tech.kayys.tafkir.data;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;

import java.io.*;
import java.nio.file.*;
import java.util.zip.GZIPInputStream;

/**
 * MNIST dataset loader.
 * Downloads and parses the classic MNIST dataset from Yann LeCun's website.
 */
public final class MNISTDataset {

    private static final String BASE_URL = "http://yann.lecun.com/exdb/mnist/";

    private final TafkirTensor images;  // [N, 784] — flattened 28x28, normalized to [0,1]
    private final TafkirTensor labels;  // [N] — class indices 0-9
    private final int numSamples;

    private MNISTDataset(TafkirTensor images, TafkirTensor labels) {
        this.images = images;
        this.labels = labels;
        this.numSamples = (int) images.shape()[0];
    }

    public TafkirTensor images() { return images; }
    public TafkirTensor labels() { return labels; }
    public int numSamples() { return numSamples; }

    /**
     * Returns a batch of images and labels.
     */
    public TafkirTensor[] batch(int start, int end) {
        if (end > numSamples) end = numSamples;
        int batchSize = end - start;

        float[] imgBatch = new float[batchSize * 784];
        float[] lblBatch = new float[batchSize];

        float[] allImages = images.data();
        float[] allLabels = labels.data();

        System.arraycopy(allImages, start * 784, imgBatch, 0, batchSize * 784);
        System.arraycopy(allLabels, start, lblBatch, 0, batchSize);

        return new TafkirTensor[]{
            TafkirTensor.of(imgBatch, batchSize, 784),
            TafkirTensor.of(lblBatch, batchSize)
        };
    }

    public static MNISTDataset loadTrain(String dataDir) throws IOException {
        return load(dataDir, "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz");
    }

    public static MNISTDataset loadTest(String dataDir) throws IOException {
        return load(dataDir, "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz");
    }

    private static MNISTDataset load(String dataDir, String imagesFile, String labelsFile) throws IOException {
        Path dir = Paths.get(dataDir);
        Files.createDirectories(dir);

        Path imagesPath = dir.resolve(imagesFile);
        Path labelsPath = dir.resolve(labelsFile);

        if (!Files.exists(imagesPath)) download(BASE_URL + imagesFile, imagesPath);
        if (!Files.exists(labelsPath)) download(BASE_URL + labelsFile, labelsPath);

        float[] images = parseImages(imagesPath);
        float[] labels = parseLabels(labelsPath);
        int numSamples = labels.length;

        for (int i = 0; i < images.length; i++) images[i] = images[i] / 255.0f;

        return new MNISTDataset(
            TafkirTensor.of(images, numSamples, 784),
            TafkirTensor.of(labels, numSamples)
        );
    }

    private static void download(String url, Path dest) throws IOException {
        System.out.println("Downloading " + url + " ...");
        try (InputStream in = new java.net.URL(url).openStream();
             OutputStream out = Files.newOutputStream(dest)) {
            in.transferTo(out);
        }
        System.out.println("Saved to " + dest);
    }

    private static float[] parseImages(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new GZIPInputStream(Files.newInputStream(path)))) {
            int magic = in.readInt();
            if (magic != 0x00000803) throw new IOException("Invalid MNIST image magic: " + magic);
            int numImages = in.readInt();
            int numRows = in.readInt();
            int numCols = in.readInt();
            if (numRows != 28 || numCols != 28) throw new IOException("Expected 28x28 images");
            float[] data = new float[numImages * 784];
            for (int i = 0; i < data.length; i++) data[i] = in.readUnsignedByte();
            return data;
        }
    }

    private static float[] parseLabels(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new GZIPInputStream(Files.newInputStream(path)))) {
            int magic = in.readInt();
            if (magic != 0x00000801) throw new IOException("Invalid MNIST label magic: " + magic);
            int numLabels = in.readInt();
            float[] data = new float[numLabels];
            for (int i = 0; i < numLabels; i++) data[i] = in.readUnsignedByte();
            return data;
        }
    }
}
