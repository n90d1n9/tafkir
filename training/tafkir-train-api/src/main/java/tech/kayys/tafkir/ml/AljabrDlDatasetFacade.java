package tech.kayys.tafkir.ml;

import tech.kayys.tafkir.ml.autograd.GradTensor;
import tech.kayys.tafkir.train.data.DataLoader;

/**
 * Dataset construction helpers inherited by {@link Aljabr.DL}.
 */
public class AljabrDlDatasetFacade extends AljabrDlLabelFacade {
    protected AljabrDlDatasetFacade() {
    }

    public static DataLoader.TensorDataset tensorDataset(GradTensor inputs, GradTensor labels) {
        return DataLoader.tensorDataset(inputs, labels);
    }

    public static DataLoader.TensorDataset classificationDataset(GradTensor inputs, int[] labels) {
        return DataLoader.classificationDataset(inputs, labels);
    }

    public static DataLoader.TensorDataset binaryDataset(GradTensor inputs, int[] labels) {
        return DataLoader.binaryDataset(inputs, labels);
    }

    public static DataLoader.TensorDataset binaryDataset(GradTensor inputs, int[][] labels) {
        return DataLoader.binaryDataset(inputs, labels);
    }

    public static DataLoader.TensorDataset binaryDataset(GradTensor inputs, boolean[][] labels) {
        return DataLoader.binaryDataset(inputs, labels);
    }

    public static DataLoader.TensorDataset binaryDataset(GradTensor inputs, float[][] labels) {
        return DataLoader.binaryDataset(inputs, labels);
    }

    public static DataLoader.TensorDataset multiLabelBinaryDataset(GradTensor inputs, int[][] labels) {
        return binaryDataset(inputs, labels);
    }

    public static DataLoader.TensorDataset multiLabelBinaryDataset(GradTensor inputs, boolean[][] labels) {
        return binaryDataset(inputs, labels);
    }

    public static DataLoader.TensorDataset multiLabelBinaryDataset(GradTensor inputs, float[][] labels) {
        return binaryDataset(inputs, labels);
    }
}
