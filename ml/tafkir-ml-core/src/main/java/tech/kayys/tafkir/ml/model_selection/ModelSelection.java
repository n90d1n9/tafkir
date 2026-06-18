package tech.kayys.tafkir.ml.model_selection;

import java.util.ArrayList;
import java.util.List;

/**
 * Model selection utilities: train/test split, k-fold cross-validation.
 */
public final class ModelSelection {

    private ModelSelection() {
    }

    /**
     * A single fold containing train and validation indices.
     */
    public static final class Fold {
        public final int[] trainIndices;
        public final int[] valIndices;

        public Fold(int[] trainIndices, int[] valIndices) {
            this.trainIndices = trainIndices;
            this.valIndices = valIndices;
        }
    }

    /**
     * K-Fold cross-validator.
     */
    public static final class KFold {
        private final int nSplits;
        private final boolean shuffle;

        public KFold(int nSplits) {
            this(nSplits, false);
        }

        public KFold(int nSplits, boolean shuffle) {
            if (nSplits < 2) {
                throw new IllegalArgumentException("nSplits must be >= 2, got: " + nSplits);
            }
            this.nSplits = nSplits;
            this.shuffle = shuffle;
        }

        public List<Fold> split(int nSamples) {
            List<Fold> folds = new ArrayList<>(nSplits);
            int foldSize = nSamples / nSplits;

            for (int k = 0; k < nSplits; k++) {
                int valStart = k * foldSize;
                int valEnd = (k == nSplits - 1) ? nSamples : valStart + foldSize;
                int valLen = valEnd - valStart;
                int trainLen = nSamples - valLen;

                int[] val = new int[valLen];
                int[] train = new int[trainLen];

                for (int i = 0; i < valLen; i++) val[i] = valStart + i;
                int ti = 0;
                for (int i = 0; i < nSamples; i++) {
                    if (i < valStart || i >= valEnd) train[ti++] = i;
                }
                folds.add(new Fold(train, val));
            }
            return folds;
        }
    }

    /**
     * Train/test split result.
     */
    public static final class Split {
        public final int[] trainIndices;
        public final int[] testIndices;

        public Split(int[] trainIndices, int[] testIndices) {
            this.trainIndices = trainIndices;
            this.testIndices = testIndices;
        }
    }

    /**
     * Split indices into train/test sets.
     *
     * @param nSamples  total sample count
     * @param testSize  fraction reserved for testing (0 &lt; testSize &lt; 1)
     * @return split containing train and test indices
     */
    public static Split trainTestSplit(int nSamples, double testSize) {
        int testLen = Math.max(1, (int) Math.round(nSamples * testSize));
        int trainLen = nSamples - testLen;
        int[] train = new int[trainLen];
        int[] test = new int[testLen];
        for (int i = 0; i < trainLen; i++) train[i] = i;
        for (int i = 0; i < testLen; i++) test[i] = trainLen + i;
        return new Split(train, test);
    }
}
