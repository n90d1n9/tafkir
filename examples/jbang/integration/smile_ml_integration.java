///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS com.github.haifengl:smile-core:3.1.1
//DEPS ${user.home}/.tafkir/jbang/libs/tafkir-sdk-nn-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.tafkir/jbang/libs/tafkir-sdk-autograd-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.tafkir/jbang/libs/tafkir-runtime-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.tafkir/jbang/libs/tafkir-spi-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.tafkir/jbang/libs/tafkir-spi-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.tafkir/jbang/libs/tafkir-spi-model-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.tafkir/jbang/libs/tafkir-sdk-api-0.1.0-SNAPSHOT.jar
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED

import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.vector.DoubleVector;
import smile.data.vector.IntVector;
import smile.math.matrix.Matrix;
import smile.stat.distribution.GaussianDistribution;

import tech.kayys.tafkir.ml.autograd.GradTensor;
import tech.kayys.tafkir.ml.nn.Linear;
import tech.kayys.tafkir.ml.nn.ReLU;
import tech.kayys.tafkir.ml.nn.Dropout;
import tech.kayys.tafkir.ml.nn.loss.CrossEntropyLoss;

import tech.kayys.tafkir.ml.nn.Trainer;

import java.util.*;
import java.util.stream.*;

/**
 * Smile ML Integration with Tafkir SDK.
 *
 * This example demonstrates how to integrate Tafkir SDK with Smile
 * (Statistical Machine Intelligence & Learning Engine), a fast and
 * comprehensive machine learning system for the JVM.
 *
 * Use Cases:
 * - Using Smile's data preprocessing and normalization
 * - Combining Smile's traditional ML algorithms with Tafkir neural networks
 * - Feature engineering with Smile, inference with Tafkir
 * - Ensemble methods combining both libraries
 * - Statistical analysis with Smile, deep learning with Tafkir
 *
 * Run: jbang smile_ml_integration.java
 */
public class smile_ml_integration {

    // ──────────────────────────────────────────────────────────────────────
    // Data Conversion Utilities
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Convert Smile DataFrame to Tafkir GradTensor.
     */
    static GradTensor dataFrameToTensor(DataFrame df) {
        double[][] data = df.toArray();
        int rows = data.length;
        int cols = data[0].length;
        
        float[] floatData = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                floatData[i * cols + j] = (float) data[i][j];
            }
        }
        
        return GradTensor.of(floatData, rows, cols);
    }

    /**
     * Convert Smile Matrix to Tafkir GradTensor.
     */
    static GradTensor matrixToTensor(Matrix matrix) {
        int rows = matrix.nrow();
        int cols = matrix.ncol();
        float[] data = new float[rows * cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * cols + j] = (float) matrix.get(i, j);
            }
        }
        
        return GradTensor.of(data, rows, cols);
    }

    /**
     * Convert Tafkir GradTensor to Smile DataFrame.
     */
    static DataFrame tensorToDataFrame(GradTensor tensor, String[] columnNames) {
        int rows = (int) tensor.shape()[0];
        int cols = (int) tensor.shape()[1];
        float[] data = tensor.data();
        
        double[][] doubleData = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                doubleData[i][j] = data[i * cols + j];
            }
        }
        
        return DataFrame.of(doubleData, columnNames);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern 1: Smile Preprocessing → Tafkir Inference
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Demonstrates using Smile's preprocessing capabilities with Tafkir models.
     * Smile provides excellent data normalization and feature scaling.
     */
    static class SmilePreprocessorTafkirModel {
        private final double[] trainMean;
        private final double[] trainStd;
        private final tech.kayys.tafkir.ml.nn.Module tafkirModel;
        private final int featureDim;

        SmilePreprocessorTafkirModel(int featureDim) {
            this.featureDim = featureDim;
            this.trainMean = new double[featureDim];
            this.trainStd = new double[featureDim];
            
            this.tafkirModel = new tech.kayys.tafkir.ml.nn.Sequential(
                new Linear(featureDim, 64),
                new ReLU(),
                new Dropout(0.2f),
                new Linear(64, 32),
                new ReLU(),
                new Linear(32, 2)
            );

            System.out.println("📦 Smile Preprocessor + Tafkir Model");
            System.out.printf("   Features: %d → 64 → 32 → 2%n", featureDim);
        }

        /**
         * Fit normalizer on training data using Smile statistics.
         */
        void fitNormalizer(float[][] trainData) {
            // Calculate mean and std using Smile-style statistics
            for (int j = 0; j < featureDim; j++) {
                double sum = 0, sumSq = 0;
                for (int i = 0; i < trainData.length; i++) {
                    sum += trainData[i][j];
                    sumSq += trainData[i][j] * trainData[i][j];
                }
                trainMean[j] = sum / trainData.length;
                double variance = (sumSq / trainData.length) - (trainMean[j] * trainMean[j]);
                trainStd[j] = Math.sqrt(variance) > 0 ? Math.sqrt(variance) : 1.0;
            }
            System.out.println("📊 Normalizer fitted (Smile-style z-score normalization)");
        }

        /**
         * Normalize data using fitted parameters.
         */
        float[][] normalize(float[][] data) {
            float[][] normalized = new float[data.length][data[0].length];
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[0].length; j++) {
                    normalized[i][j] = (float) ((data[i][j] - trainMean[j]) / trainStd[j]);
                }
            }
            return normalized;
        }

        /**
         * Train the Tafkir model on normalized data.
         */
        void train(float[][] trainData, int[] labels, int epochs, int batchSize) {
            // Fit normalizer
            fitNormalizer(trainData);
            
            // Normalize data
            float[][] normalized = normalize(trainData);
            
            // Convert to tensors
            float[] featureData = new float[normalized.length * featureDim];
            for (int i = 0; i < normalized.length; i++) {
                System.arraycopy(normalized[i], 0, featureData, i * featureDim, featureDim);
            }
            GradTensor inputTensor = GradTensor.of(featureData, normalized.length, featureDim);
            float[] labelData = new float[labels.length];
            for (int i = 0; i < labels.length; i++) {
                labelData[i] = labels[i];
            }
            GradTensor labelTensor = GradTensor.of(labelData, labels.length);

            // Train
            System.out.println("🏋️ Training Tafkir model...");
            var trainer = Trainer.builder()
                    .model(tafkirModel)
                    .optimizer(new tech.kayys.tafkir.ml.nn.optim.Adam(tafkirModel.parameters(), 0.001f))
                    .lossFunction((pred, target) -> new CrossEntropyLoss().compute(pred, target))
                    .epochs(epochs)
                    .callback(new Trainer.TrainingCallback() {
                        @Override
                        public void onEpochEnd(int epoch, int totalEpochs, float avgLoss) {
                            if ((epoch + 1) % 20 == 0 || epoch == 0) {
                                System.out.printf("  Epoch %3d/%d │ loss: %.4f%n", epoch + 1, totalEpochs, avgLoss);
                            }
                        }
                    })
                    .build();

            trainer.fit(inputTensor, labelTensor, batchSize);
            System.out.println("✅ Training complete!");
        }

        /**
         * Predict using normalized input.
         */
        int predict(float[] input) {
            // Normalize
            float[] normalized = new float[input.length];
            for (int i = 0; i < input.length; i++) {
                normalized[i] = (float) ((input[i] - trainMean[i]) / trainStd[i]);
            }

            // Inference
            tafkirModel.eval();
            GradTensor inputTensor = GradTensor.of(normalized, 1, featureDim);
            GradTensor output = tafkirModel.forward(inputTensor);
            float[] scores = output.data();

            // Argmax
            int maxIdx = 0;
            float maxScore = scores[0];
            for (int i = 1; i < scores.length; i++) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }


    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern 2: Smile Feature Selection → Tafkir Classification
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Uses Smile's feature importance ranking to select top features,
     * then trains a Tafkir model on the selected features.
     */
    static class SmileFeatureSelectionTafkirClassifier {
        private final RandomForest forest;
        private final tech.kayys.tafkir.ml.nn.Module tafkirClassifier;
        private final int[] selectedFeatureIndices;
        private final int originalFeatureDim;
        private final int selectedFeatureDim;

        SmileFeatureSelectionTafkirClassifier(int originalFeatureDim, int numFeaturesToSelect) {
            this.originalFeatureDim = originalFeatureDim;
            this.selectedFeatureDim = numFeaturesToSelect;
            this.selectedFeatureIndices = new int[numFeaturesToSelect];
            
            // Initialize Random Forest for feature selection
            this.forest = null; // Will be created during training
            
            // Tafkir classifier
            this.tafkirClassifier = new tech.kayys.tafkir.ml.nn.Sequential(
                new Linear(numFeaturesToSelect, 32),
                new ReLU(),
                new Linear(32, 2)
            );

            System.out.println("📦 Smile Feature Selection + Tafkir Classifier");
            System.out.printf("   Feature selection: %d → %d (Random Forest importance)%n", 
                    originalFeatureDim, numFeaturesToSelect);
            System.out.printf("   Classifier: %d → 32 → 2%n", numFeaturesToSelect);
        }

        /**
         * Train Random Forest, select top features, then train Tafkir.
         */
        void train(float[][] trainData, int[] labels, int epochs, int batchSize) {
            System.out.println("\n━━━ Pattern: Smile Feature Selection → Tafkir Classification ━━━");

            // Convert to double for Smile
            double[][] doubleData = new double[trainData.length][trainData[0].length];
            for (int i = 0; i < trainData.length; i++) {
                for (int j = 0; j < trainData[0].length; j++) {
                    doubleData[i][j] = trainData[i][j];
                }
            }

            // Step 1: Train Random Forest for feature importance
            System.out.println("🌲 Training Random Forest for feature importance...");
            DataFrame df = DataFrame.of(doubleData);
            df = df.merge(IntVector.of("y", labels));
            
            RandomForest rf = RandomForest.fit(Formula.lhs("y"), df);
            
            // Get feature importance
            double[] importance = rf.importance();
            
            // Select top features
            Integer[] indices = IntStream.range(0, originalFeatureDim)
                    .boxed()
                    .toArray(Integer[]::new);
            Arrays.sort(indices, (a, b) -> Double.compare(importance[b], importance[a]));
            
            for (int i = 0; i < selectedFeatureDim; i++) {
                selectedFeatureIndices[i] = indices[i];
            }
            
            System.out.printf("✅ Selected top %d features (from %d)%n", 
                    selectedFeatureDim, originalFeatureDim);
            System.out.printf("   Top feature indices: %s%n", 
                    Arrays.toString(selectedFeatureIndices));

            // Step 2: Extract selected features
            float[][] selectedData = new float[trainData.length][selectedFeatureDim];
            for (int i = 0; i < trainData.length; i++) {
                for (int j = 0; j < selectedFeatureDim; j++) {
                    selectedData[i][j] = trainData[i][selectedFeatureIndices[j]];
                }
            }

            // Step 3: Train Tafkir classifier
            System.out.println("🏋️ Training Tafkir classifier on selected features...");
            float[] featureData = new float[selectedData.length * selectedFeatureDim];
            for (int i = 0; i < selectedData.length; i++) {
                System.arraycopy(selectedData[i], 0, featureData, i * selectedFeatureDim, selectedFeatureDim);
            }
            GradTensor inputTensor = GradTensor.of(featureData, selectedData.length, selectedFeatureDim);
            
            float[] labelData = new float[labels.length];
            for (int i = 0; i < labels.length; i++) {
                labelData[i] = labels[i];
            }
            GradTensor labelTensor = GradTensor.of(labelData, labels.length);

            var trainer = Trainer.builder()
                    .model(tafkirClassifier)
                    .optimizer(new tech.kayys.tafkir.ml.nn.optim.Adam(tafkirClassifier.parameters(), 0.001f))
                    .lossFunction((pred, target) -> new CrossEntropyLoss().compute(pred, target))
                    .epochs(epochs)
                    .callback(new Trainer.TrainingCallback() {
                        @Override
                        public void onEpochEnd(int epoch, int totalEpochs, float avgLoss) {
                            if ((epoch + 1) % 20 == 0 || epoch == 0) {
                                System.out.printf("  Epoch %3d/%d │ loss: %.4f%n", epoch + 1, totalEpochs, avgLoss);
                            }
                        }
                    })
                    .build();

            trainer.fit(inputTensor, labelTensor, batchSize);
            System.out.println("✅ Training complete!");
        }

        /**
         * Predict using selected features.
         */
        int predict(float[] input) {
            // Select features
            float[] selected = new float[selectedFeatureDim];
            for (int i = 0; i < selectedFeatureDim; i++) {
                selected[i] = input[selectedFeatureIndices[i]];
            }

            // Inference
            tafkirClassifier.eval();
            GradTensor inputTensor = GradTensor.of(selected, 1, selectedFeatureDim);
            GradTensor output = tafkirClassifier.forward(inputTensor);
            float[] scores = output.data();

            // Argmax
            int maxIdx = 0;
            float maxScore = scores[0];
            for (int i = 1; i < scores.length; i++) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }


    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern 3: Ensemble (Smile + Tafkir)
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Ensemble classifier combining Smile's Random Forest with Tafkir neural network.
     * Uses voting or probability averaging for final prediction.
     */
    static class EnsembleSmileTafkir {
        private RandomForest smileForest;
        private final tech.kayys.tafkir.ml.nn.Module tafkirModel;
        private final int featureDim;
        private final double smileWeight;
        private final double tafkirWeight;

        EnsembleSmileTafkir(int featureDim, double smileWeight, double tafkirWeight) {
            this.featureDim = featureDim;
            this.smileWeight = smileWeight;
            this.tafkirWeight = tafkirWeight;
            
            this.tafkirModel = new tech.kayys.tafkir.ml.nn.Sequential(
                new Linear(featureDim, 64),
                new ReLU(),
                new Linear(64, 32),
                new ReLU(),
                new Linear(32, 2)
            );

            System.out.println("🎭 Ensemble: Smile Random Forest + Tafkir Neural Network");
            System.out.printf("   Weights: Smile=%.1f%%, Tafkir=%.1f%%%n", 
                    smileWeight * 100, tafkirWeight * 100);
        }

        /**
         * Train both models.
         */
        void train(float[][] trainData, int[] labels, int epochs, int batchSize) {
            System.out.println("\n━━━ Pattern: Ensemble (Smile + Tafkir) ━━━");

            // Convert to double for Smile
            double[][] doubleData = new double[trainData.length][trainData[0].length];
            for (int i = 0; i < trainData.length; i++) {
                for (int j = 0; j < trainData[0].length; j++) {
                    doubleData[i][j] = trainData[i][j];
                }
            }
            DataFrame df = DataFrame.of(doubleData);
            df = df.merge(IntVector.of("y", labels));

            // Train Smile Random Forest
            System.out.println("🌲 Training Smile Random Forest...");
            smileForest = RandomForest.fit(Formula.lhs("y"), df);
            System.out.println("✅ Random Forest trained!");

            // Train Tafkir model
            System.out.println("🏋️ Training Tafkir Neural Network...");
            float[] featureData = new float[trainData.length * featureDim];
            for (int i = 0; i < trainData.length; i++) {
                System.arraycopy(trainData[i], 0, featureData, i * featureDim, featureDim);
            }
            GradTensor inputTensor = GradTensor.of(featureData, trainData.length, featureDim);
            float[] labelData = new float[labels.length];
            for (int i = 0; i < labels.length; i++) {
                labelData[i] = labels[i];
            }
            GradTensor labelTensor = GradTensor.of(labelData, labels.length);

            var trainer = Trainer.builder()
                    .model(tafkirModel)
                    .optimizer(new tech.kayys.tafkir.ml.nn.optim.Adam(tafkirModel.parameters(), 0.001f))
                    .lossFunction((pred, target) -> new CrossEntropyLoss().compute(pred, target))
                    .epochs(epochs)
                    .callback(new Trainer.TrainingCallback() {
                        @Override
                        public void onEpochEnd(int epoch, int totalEpochs, float avgLoss) {
                            if ((epoch + 1) % 20 == 0 || epoch == 0) {
                                System.out.printf("  Epoch %3d/%d │ loss: %.4f%n", epoch + 1, totalEpochs, avgLoss);
                            }
                        }
                    })
                    .build();

            trainer.fit(inputTensor, labelTensor, batchSize);
            System.out.println("✅ Neural Network trained!");
        }

        /**
         * Predict using ensemble voting.
         */
        EnsembleResult predict(float[] input) {
            // Smile prediction
            double[] smileProbs = new double[2];
            int label = smileForest.predict(dfOf(input).apply(0));
            smileProbs[label] = 1.0; // RandomForest.predict(Tuple) returns int in Smile 3 for classification
            
            // Tafkir prediction
            tafkirModel.eval();
            GradTensor inputTensor = GradTensor.of(input, 1, featureDim);
            GradTensor output = tafkirModel.forward(inputTensor);
            float[] tafkirScores = output.data();
            
            // Softmax for Tafkir
            float maxScore = Math.max(tafkirScores[0], tafkirScores[1]);
            float exp0 = (float) Math.exp(tafkirScores[0] - maxScore);
            float exp1 = (float) Math.exp(tafkirScores[1] - maxScore);
            float sum = exp0 + exp1;
            float[] tafkirProbs = new float[]{exp0 / sum, exp1 / sum};

            // Ensemble: weighted average
            double ensembleProb0 = smileWeight * smileProbs[0] + tafkirWeight * tafkirProbs[0];
            double ensembleProb1 = smileWeight * smileProbs[1] + tafkirWeight * tafkirProbs[1];
            
            int ensembleClass = ensembleProb0 > ensembleProb1 ? 0 : 1;
            int smileClass = smileProbs[0] > smileProbs[1] ? 0 : 1;
            int tafkirClass = tafkirProbs[0] > tafkirProbs[1] ? 0 : 1;

            return new EnsembleResult(ensembleClass, ensembleProb0, ensembleProb1, 
                    smileClass, tafkirClass, smileProbs, tafkirProbs);
        }

        // Helper to create DataFrame from single sample
        private DataFrame dfOf(float[] input) {
            double[][] data = new double[1][input.length];
            for (int i = 0; i < input.length; i++) {
                data[0][i] = input[i];
            }
            return DataFrame.of(data);
        }


    }

    static class EnsembleResult {
        final int ensembleClass;
        final double ensembleProb0;
        final double ensembleProb1;
        final int smileClass;
        final int tafkirClass;
        final double[] smileProbs;
        final float[] tafkirProbs;

        EnsembleResult(int ensembleClass, double ensembleProb0, double ensembleProb1,
                      int smileClass, int tafkirClass, double[] smileProbs, float[] tafkirProbs) {
            this.ensembleClass = ensembleClass;
            this.ensembleProb0 = ensembleProb0;
            this.ensembleProb1 = ensembleProb1;
            this.smileClass = smileClass;
            this.tafkirClass = tafkirClass;
            this.smileProbs = smileProbs;
            this.tafkirProbs = tafkirProbs;
        }

        @Override
        public String toString() {
            return String.format("Ensemble=%d (%.1f%%), Smile=%d, Tafkir=%d",
                    ensembleClass, Math.max(ensembleProb0, ensembleProb1) * 100,
                    smileClass, tafkirClass);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern 4: Smile Statistical Features → Tafkir
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Extract statistical features using Smile's distribution utilities,
     * then classify with Tafkir.
     */
    static class StatisticalFeatureExtractor {
        private final tech.kayys.tafkir.ml.nn.Module classifier;
        private final int featureDim;

        StatisticalFeatureExtractor(int baseFeatures, int statFeatures) {
            this.featureDim = baseFeatures + statFeatures;
            
            this.classifier = new tech.kayys.tafkir.ml.nn.Sequential(
                new Linear(featureDim, 32),
                new ReLU(),
                new Linear(32, 2)
            );

            System.out.println("📊 Statistical Feature Extractor + Tafkir");
            System.out.printf("   Base features: %d, Statistical features: %d, Total: %d%n",
                    baseFeatures, statFeatures, featureDim);
        }

        /**
         * Extract statistical features from raw data.
         */
        float[] extractFeatures(float[] rawData) {
            // Calculate statistics using Smile-style approach
            int n = rawData.length;
            
            // Mean
            double sum = 0;
            for (float v : rawData) sum += v;
            double mean = sum / n;
            
            // Variance and Std
            double sumSq = 0;
            for (float v : rawData) sumSq += (v - mean) * (v - mean);
            double variance = sumSq / n;
            double std = Math.sqrt(variance);
            
            // Min, Max
            float min = rawData[0], max = rawData[0];
            for (float v : rawData) {
                if (v < min) min = v;
                if (v > max) max = v;
            }
            
            // Skewness (using Gaussian approximation)
            double sumCube = 0;
            for (float v : rawData) sumCube += Math.pow((v - mean) / std, 3);
            double skewness = n > 2 ? (sumCube / n) * (Math.sqrt(n * (n - 1.0)) / (n - 2.0)) : 0;
            
            // Kurtosis
            double sumQuad = 0;
            for (float v : rawData) sumQuad += Math.pow((v - mean) / std, 4);
            double kurtosis = n > 3 ? (sumQuad / n) - 3 : 0; // Excess kurtosis

            // Combine raw data with statistical features
            float[] features = new float[rawData.length + 6];
            System.arraycopy(rawData, 0, features, 0, rawData.length);
            features[rawData.length] = (float) mean;
            features[rawData.length + 1] = (float) std;
            features[rawData.length + 2] = (float) min;
            features[rawData.length + 3] = (float) max;
            features[rawData.length + 4] = (float) skewness;
            features[rawData.length + 5] = (float) kurtosis;

            return features;
        }

        void train(float[][] trainData, int[] labels, int epochs, int batchSize) {
            System.out.println("\n━━━ Pattern: Statistical Features + Tafkir ━━━");
            System.out.println("📊 Extracting statistical features...");

            // Extract features
            float[][] extractedData = new float[trainData.length][featureDim];
            for (int i = 0; i < trainData.length; i++) {
                extractedData[i] = extractFeatures(trainData[i]);
            }

            // Train
            float[] featureData = new float[extractedData.length * featureDim];
            for (int i = 0; i < extractedData.length; i++) {
                System.arraycopy(extractedData[i], 0, featureData, i * featureDim, featureDim);
            }
            GradTensor inputTensor = GradTensor.of(featureData, extractedData.length, featureDim);
            float[] labelData = new float[labels.length];
            for (int i = 0; i < labels.length; i++) {
                labelData[i] = labels[i];
            }
            GradTensor labelTensor = GradTensor.of(labelData, labels.length);

            var trainer = Trainer.builder()
                    .model(classifier)
                    .optimizer(new tech.kayys.tafkir.ml.nn.optim.Adam(classifier.parameters(), 0.001f))
                    .lossFunction((pred, target) -> new CrossEntropyLoss().compute(pred, target))
                    .epochs(epochs)
                    .build();

            trainer.fit(inputTensor, labelTensor, batchSize);
            System.out.println("✅ Training complete!");
        }

        int predict(float[] rawData) {
            float[] features = extractFeatures(rawData);
            classifier.eval();
            GradTensor input = GradTensor.of(features, 1, featureDim);
            GradTensor output = classifier.forward(input);
            float[] scores = output.data();

            return scores[0] > scores[1] ? 0 : 1;
        }


    }

    // ──────────────────────────────────────────────────────────────────────
    // Demo Dataset Generator
    // ──────────────────────────────────────────────────────────────────────

    static class DemoDataset {
        static float[][] generateData(int numSamples, int numFeatures, Random rand) {
            float[][] data = new float[numSamples][numFeatures];
            for (int i = 0; i < numSamples; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    data[i][j] = (float) rand.nextGaussian();
                }
            }
            return data;
        }

        static int[] generateLabels(int numSamples, float[][] data, Random rand) {
            int[] labels = new int[numSamples];
            for (int i = 0; i < numSamples; i++) {
                // Simple rule: if sum of first half > sum of second half, class 1
                float sum1 = 0, sum2 = 0;
                int mid = data[i].length / 2;
                for (int j = 0; j < mid; j++) sum1 += data[i][j];
                for (int j = mid; j < data[i].length; j++) sum2 += data[i][j];
                labels[i] = (sum1 > sum2) ? 1 : 0;
            }
            return labels;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Main: Run All Integration Examples
    // ──────────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║   Tafkir SDK + Smile ML Integration Examples            ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        Random rand = new Random(42);
        int NUM_SAMPLES = 200;
        int NUM_FEATURES = 20;
        int EPOCHS = 50;

        // Generate dataset
        System.out.printf("%n📚 Generating dataset: %d samples, %d features%n", NUM_SAMPLES, NUM_FEATURES);
        float[][] trainData = DemoDataset.generateData(NUM_SAMPLES, NUM_FEATURES, rand);
        int[] trainLabels = DemoDataset.generateLabels(NUM_SAMPLES, trainData, rand);

        // Test data
        float[][] testData = DemoDataset.generateData(20, NUM_FEATURES, rand);
        int[] testLabels = DemoDataset.generateLabels(20, testData, rand);

        try {
            // Pattern 1: Smile Preprocessing → Tafkir Inference
            System.out.println("\n" + "═".repeat(60));
            System.out.println("PATTERN 1: Smile Preprocessing → Tafkir Inference");
            System.out.println("═".repeat(60));
            
            var pattern1 = new SmilePreprocessorTafkirModel(NUM_FEATURES);
            pattern1.train(trainData, trainLabels, EPOCHS, 16);

            // Evaluate
            int correct1 = 0;
            for (int i = 0; i < testData.length; i++) {
                if (pattern1.predict(testData[i]) == testLabels[i]) correct1++;
            }
            System.out.printf("\n📈 Test Accuracy: %d/%d (%.1f%%)%n", 
                    correct1, testData.length, 100.0 * correct1 / testData.length);

            // Pattern 2: Smile Feature Selection → Tafkir Classification
            System.out.println("\n" + "═".repeat(60));
            System.out.println("PATTERN 2: Smile Feature Selection → Tafkir Classification");
            System.out.println("═".repeat(60));
            
            var pattern2 = new SmileFeatureSelectionTafkirClassifier(NUM_FEATURES, 10);
            pattern2.train(trainData, trainLabels, EPOCHS, 16);

            // Evaluate
            int correct2 = 0;
            for (int i = 0; i < testData.length; i++) {
                if (pattern2.predict(testData[i]) == testLabels[i]) correct2++;
            }
            System.out.printf("\n📈 Test Accuracy (with feature selection): %d/%d (%.1f%%)%n", 
                    correct2, testData.length, 100.0 * correct2 / testData.length);

            // Pattern 3: Ensemble (Smile + Tafkir)
            System.out.println("\n" + "═".repeat(60));
            System.out.println("PATTERN 3: Ensemble (Smile Random Forest + Tafkir NN)");
            System.out.println("═".repeat(60));
            
            var pattern3 = new EnsembleSmileTafkir(NUM_FEATURES, 0.5, 0.5);
            pattern3.train(trainData, trainLabels, EPOCHS, 16);

            // Evaluate with detailed results
            System.out.println("\n📈 Testing ensemble predictions:");
            int correct3 = 0;
            for (int i = 0; i < Math.min(5, testData.length); i++) {
                EnsembleResult result = pattern3.predict(testData[i]);
                boolean isCorrect = result.ensembleClass == testLabels[i];
                if (isCorrect) correct3++;
                System.out.printf("  Sample %d: %s (Actual: %d) %s%n", 
                        i + 1, result, testLabels[i], isCorrect ? "✓" : "✗");
            }
            System.out.printf("\n📈 Test Accuracy (Ensemble): %d/%d (%.1f%%)%n", 
                    correct3, Math.min(5, testData.length), 100.0 * correct3 / Math.min(5, testData.length));

            // Pattern 4: Statistical Features
            System.out.println("\n" + "═".repeat(60));
            System.out.println("PATTERN 4: Smile Statistical Features → Tafkir");
            System.out.println("═".repeat(60));
            
            var pattern4 = new StatisticalFeatureExtractor(NUM_FEATURES, 6);
            pattern4.train(trainData, trainLabels, EPOCHS, 16);

            // Evaluate
            int correct4 = 0;
            for (int i = 0; i < testData.length; i++) {
                if (pattern4.predict(testData[i]) == testLabels[i]) correct4++;
            }
            System.out.printf("\n📈 Test Accuracy (with statistical features): %d/%d (%.1f%%)%n", 
                    correct4, testData.length, 100.0 * correct4 / testData.length);

            System.out.println("\n╔══════════════════════════════════════════════════════════╗");
            System.out.println("║   ✅ All Smile ML integration examples completed!       ║");
            System.out.println("╚══════════════════════════════════════════════════════════╝");

        } catch (Exception e) {
            System.err.println("❌ Error during integration demo: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
