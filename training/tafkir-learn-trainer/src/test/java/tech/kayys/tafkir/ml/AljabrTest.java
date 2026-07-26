package tech.kayys.tafkir.ml;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Disabled;
import tech.kayys.tafkir.ml.autograd.GradTensor;
import tech.kayys.tafkir.ml.nn.NNModule;
import tech.kayys.tafkir.ml.nn.layer.Linear;
import tech.kayys.tafkir.ml.train.TrainingReport;
import tech.kayys.tafkir.ml.train.TrainingReportAdvisor;
import tech.kayys.tafkir.ml.train.TrainingReportComparisonActionPlanArtifacts;
import tech.kayys.tafkir.ml.train.TrainingReportComparisonActionPlanExport;
// TODO: nlp package - import tech.kayys.tafkir.ml.nlp.Pipeline;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.tafkir.train.data.DataLoader;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public class AljabrTest {

    @Test
    public void testTensorCreation() {
        GradTensor t = Aljabr.tensor(new float[]{1, 2, 3}, 1, 3);
        assertNotNull(t);
        assertEquals(3, t.numel());
    }

    @Test
    public void testNnUsage() {
        // Use constructor directly as shown in Aljabr.java examples
        Linear linear = new Linear(10, 20);
        assertNotNull(linear);
    }

    @Disabled("NLP package not yet implemented")
    @Test
    public void testPipelineFactory() {
        // Use an unknown task to trigger PipelineException
        // assertThrows(tech.kayys.tafkir.ml.nlp.PipelineException.class, () -> 
        //     Aljabr.pipeline("unknown-task", "any"));
    }

    @Test
    public void testDeviceHeuristics() {
        DeviceType device = Aljabr.defaultDevice();
        assertNotNull(device);
        // On most CI/Local it will be CPU
        System.out.println("Default device: " + device);
    }

    @Test
    public void testComparisonActionPlanFacadeWritesAndVerifiesArtifacts() throws IOException {
        TrainingReport baseline = TrainingReport.of(runtimeProfileReport(2_000_000L, 5_000_000L));
        TrainingReport candidate = TrainingReport.of(runtimeProfileReport(4_000_000L, 10_000_000L));

        assertEquals(TrainingReportAdvisor.COMPARISON_ACTION_PLAN_SCHEMA,
                Aljabr.DL.trainingReportComparisonActionPlanSchema());
        Map<String, Object> actionPlan = Aljabr.DL.trainingReportComparisonActionPlan(baseline, candidate);
        assertEquals(TrainingReportComparisonActionPlanExport.SCHEMA, actionPlan.get("schema"));

        TrainingReportComparisonActionPlanExport export =
                Aljabr.DL.trainingReportComparisonActionPlanExport(baseline, candidate);
        assertTrue(export.regressed());
        assertTrue(export.requiresAttention());
        assertTrue(Aljabr.DL.trainingReportComparisonActionPlanMarkdown(baseline, candidate)
                .contains("Runtime Regression Summary"));

        Path directory = Files.createTempDirectory("aljabr-action-plan-facade");
        TrainingReportComparisonActionPlanArtifacts.ArtifactBundle bundle =
                Aljabr.DL.writeTrainingReportComparisonActionPlanArtifacts(directory, export);
        TrainingReportComparisonActionPlanArtifacts.Verification verification =
                Aljabr.DL.verifyTrainingReportComparisonActionPlanArtifacts(bundle);
        assertTrue(verification.passed());
        assertTrue(Aljabr.DL.trainingReportComparisonActionPlanArtifactVerificationMarkdown(verification)
                .contains("**Status:** `PASS`"));
        assertEquals(export.markdown(),
                Aljabr.DL.readTrainingReportComparisonActionPlanMarkdown(bundle.markdownFile()));
        assertEquals(export.runtimeRegression().toMap(),
                Aljabr.DL.readTrainingReportComparisonActionPlanExport(bundle.jsonFile())
                        .runtimeRegression()
                        .toMap());

        Files.deleteIfExists(bundle.jsonFile());
        Files.deleteIfExists(bundle.markdownFile());
        Files.deleteIfExists(bundle.manifestFile());
        Files.deleteIfExists(bundle.directory());
    }

    @Test
    public void testTrainValidationTestSplitFacade() {
        DataLoader.TensorDatasetThreeWaySplit split = Aljabr.DL.classificationStratifiedTrainValidationTestSplit(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0.8f, 0.2f,
                        0.7f, 0.3f,
                        0f, 1f,
                        0.1f, 0.9f,
                        0.2f, 0.8f,
                        0.3f, 0.7f
                }, 8, 2),
                new int[] {0, 0, 0, 0, 1, 1, 1, 1},
                0.5,
                0.25,
                7L);

        assertEquals(4, split.train().size());
        assertEquals(2, split.validation().size());
        assertEquals(2, split.test().size());
        assertEquals(2, split.trainLoader(2).numBatches());
        assertEquals(1, split.validationLoader(2).numBatches());
        assertEquals(1, split.testLoader(2).numBatches());
    }

    @Test
    public void testClassBalancedClassificationLoaderFacade() {
        DataLoader.TensorDataLoader loader = Aljabr.DL.classBalancedClassificationLoader(
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f,
                        7f,
                        8f
                }, 8, 1),
                new int[] {0, 0, 0, 0, 0, 0, 1, 1},
                4,
                2,
                2026L);

        assertEquals(2, loader.numBatches());
        for (DataLoader.Batch batch : loader) {
            int zeros = 0;
            int ones = 0;
            for (float label : batch.labels().data()) {
                if (label == 0f) {
                    zeros++;
                } else if (label == 1f) {
                    ones++;
                }
            }
            assertEquals(2, zeros);
            assertEquals(2, ones);
        }
    }

    @Test
    public void testStratifiedClassificationLoaderFacade() {
        int[] labels = {0, 0, 0, 0, 0, 0, 1, 1};
        DataLoader.TensorDataLoader loader = Aljabr.DL.stratifiedClassificationLoader(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 8, 1),
                labels,
                4,
                2,
                2032L);

        assertEquals(2, loader.numBatches());
        assertArrayEquals(new int[] {6, 2}, Aljabr.DL.stratifiedBatchSampler(
                labels, 4, 2, false, 2032L).classCounts());
        DataLoader.ClassificationDistributionReport report = Aljabr.DL.classificationDistribution(loader, 2);
        assertArrayEquals(new int[] {6, 2}, report.classCounts());
        assertArrayEquals(new double[] {0.75, 0.25}, report.classFractions(), 1e-6);
        DataLoader.ClassificationDistributionReport inferredReport = Aljabr.DL.classificationDistribution(loader);
        assertEquals(2, inferredReport.numClasses());
        assertArrayEquals(report.classCounts(), inferredReport.classCounts());
        for (DataLoader.Batch batch : loader) {
            int zeros = 0;
            int ones = 0;
            for (float label : batch.labels().data()) {
                if (label == 0f) {
                    zeros++;
                } else if (label == 1f) {
                    ones++;
                }
            }
            assertEquals(3, zeros);
            assertEquals(1, ones);
        }
    }

    @Test
    public void testWeightedSingleLabelLoaderFacades() {
        int[] labels = {0, 0, 0, 1};

        assertArrayEquals(
                new float[] {2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f},
                Aljabr.DL.classBalancedSampleWeights(labels),
                1e-6f);
        assertArrayEquals(
                new float[] {2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f},
                Aljabr.DL.binaryBalancedSampleWeights(labels),
                1e-6f);

        DataLoader.TensorDataLoader classification = Aljabr.DL.classWeightedClassificationLoader(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                labels,
                3,
                12,
                17L);
        DataLoader.TensorDataLoader binary = Aljabr.DL.weightedBinaryLoader(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                labels,
                2,
                10,
                23L);

        assertEquals(12, classification.sampleCount());
        assertEquals(4, classification.numBatches());
        assertTrue(flattenInputs(classification).contains(4f));
        assertEquals(10, binary.sampleCount());
        assertEquals(5, binary.numBatches());
        assertTrue(flattenInputs(binary).contains(4f));
    }

    @Test
    public void testMultiLabelBalancedBinaryLoaderFacade() {
        int[][] labels = {
                {1, 0, 0},
                {1, 0, 0},
                {0, 1, 0},
                {0, 1, 0},
                {0, 0, 1},
                {1, 1, 1}
        };
        DataLoader.TensorDataLoader loader = Aljabr.DL.multiLabelBalancedBinaryLoader(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                labels,
                3,
                3,
                2026L);

        assertEquals(3, loader.numBatches());
        DataLoader.MultiLabelDistributionReport report = Aljabr.DL.multiLabelDistribution(loader, 3);
        assertEquals(9, report.sampleCount());
        assertTrue(report.positiveCount(0) >= 3);
        assertTrue(report.positiveCount(1) >= 3);
        assertTrue(report.positiveCount(2) >= 3);
        DataLoader.MultiLabelDistributionReport inferredReport = Aljabr.DL.multiLabelDistribution(loader);
        assertEquals(3, inferredReport.labelCount());
        assertArrayEquals(report.positiveCounts(), inferredReport.positiveCounts());
        for (DataLoader.Batch batch : loader) {
            int[] counts = positiveCounts(batch, 3);
            assertTrue(counts[0] >= 1);
            assertTrue(counts[1] >= 1);
            assertTrue(counts[2] >= 1);
        }
    }

    @Test
    public void testMultiLabelWeightedBinaryLoaderFacade() {
        int[][] labels = {
                {1, 0, 0},
                {1, 0, 0},
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1},
                {0, 0, 0}
        };

        assertArrayEquals(new float[] {
                2.0f / 3.0f,
                2.0f / 3.0f,
                2.0f / 3.0f,
                2.0f,
                2.0f,
                2.0f / 3.0f
        }, Aljabr.DL.multiLabelBalancedSampleWeights(labels), 1e-6f);

        DataLoader.TensorDataLoader loader = Aljabr.DL.multiLabelWeightedBinaryLoader(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                labels,
                2,
                6,
                2028L);

        assertEquals(6, loader.sampleCount());
        assertEquals(3, loader.numBatches());
        java.util.List<Float> drawn = flattenInputs(loader);
        assertTrue(drawn.contains(4f) || drawn.contains(5f));
    }

    @Test
    public void testMultiLabelStratifiedTrainValidationTestSplitFacade() {
        DataLoader.TensorDatasetThreeWaySplit split = Aljabr.DL.multiLabelBinaryStratifiedTrainValidationTestSplit(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0f, 1f,
                        0.1f, 0.9f,
                        1f, 1f,
                        0.9f, 0.9f,
                        1f, 0.5f,
                        0.9f, 0.4f,
                        0.5f, 1f,
                        0.4f, 0.9f,
                        0.6f, 0.6f,
                        0.7f, 0.7f
                }, 12, 2),
                new int[][] {
                        {1, 0, 0},
                        {1, 0, 0},
                        {0, 1, 0},
                        {0, 1, 0},
                        {0, 0, 1},
                        {0, 0, 1},
                        {1, 1, 0},
                        {1, 1, 0},
                        {1, 0, 1},
                        {1, 0, 1},
                        {0, 1, 1},
                        {0, 1, 1}
                },
                0.5,
                0.25,
                8L);

        assertEquals(6, split.train().size());
        assertEquals(3, split.validation().size());
        assertEquals(3, split.test().size());
    }

    @Test
    public void testMultiLabelStratifiedGroupTrainValidationSplitFacade() {
        int[] groups = {10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60};
        DataLoader.TensorDatasetSplit split = Aljabr.DL.multiLabelBinaryStratifiedGroupTrainValidationSplit(
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f,
                        7f,
                        8f,
                        9f,
                        10f,
                        11f,
                        12f
                }, 12, 1),
                balancedGroupedMultiLabelRows(),
                groups,
                0.5,
                43L);

        assertEquals(6, split.train().size());
        assertEquals(6, split.validation().size());
        assertArrayEquals(new int[] {3, 3, 3}, positiveCounts(split.train(), 3));
        assertArrayEquals(new int[] {3, 3, 3}, positiveCounts(split.validation(), 3));
        for (int group : groupsForRows(split.validation(), groups)) {
            assertTrue(!groupsForRows(split.train(), groups).contains(group));
        }
    }

    @Test
    public void testMultiLabelStratifiedGroupTrainValidationTestSplitFacade() {
        int[] groups = {10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60};
        DataLoader.TensorDatasetThreeWaySplit split = Aljabr.DL.multiLabelBinaryStratifiedGroupTrainValidationTestSplit(
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f,
                        7f,
                        8f,
                        9f,
                        10f,
                        11f,
                        12f
                }, 12, 1),
                balancedGroupedMultiLabelRows(),
                groups,
                0.5,
                1.0 / 6.0,
                43L);

        assertEquals(6, split.train().size());
        assertEquals(2, split.validation().size());
        assertEquals(4, split.test().size());
        assertArrayEquals(new int[] {3, 3, 3}, positiveCounts(split.train(), 3));
        assertArrayEquals(new int[] {1, 1, 1}, positiveCounts(split.validation(), 3));
        assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(split.test(), 3));
        for (int group : groupsForRows(split.validation(), groups)) {
            assertTrue(!groupsForRows(split.train(), groups).contains(group));
        }
        for (int group : groupsForRows(split.test(), groups)) {
            assertTrue(!groupsForRows(split.train(), groups).contains(group));
            assertTrue(!groupsForRows(split.validation(), groups).contains(group));
        }
    }

    @Test
    public void testMultiLabelStratifiedCrossValidationFacade() {
        var folds = Aljabr.DL.multiLabelBinaryStratifiedCrossValidationFolds(
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f,
                        7f,
                        8f,
                        9f,
                        10f,
                        11f,
                        12f
                }, 12, 1),
                new int[][] {
                        {1, 0, 0},
                        {1, 0, 0},
                        {0, 1, 0},
                        {0, 1, 0},
                        {0, 0, 1},
                        {0, 0, 1},
                        {1, 1, 0},
                        {1, 1, 0},
                        {1, 0, 1},
                        {1, 0, 1},
                        {0, 1, 1},
                        {0, 1, 1}
                },
                3,
                42L);

        assertEquals(3, folds.size());
        for (DataLoader.TensorDatasetFold fold : folds) {
            assertEquals(8, fold.train().size());
            assertEquals(4, fold.validation().size());
            assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(fold.validation(), 3));
        }
    }

    @Test
    public void testMultiLabelStratifiedGroupCrossValidationFacade() {
        int[] groups = {10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60};
        var folds = Aljabr.DL.multiLabelBinaryStratifiedGroupCrossValidationFolds(
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f,
                        7f,
                        8f,
                        9f,
                        10f,
                        11f,
                        12f
                }, 12, 1),
                new int[][] {
                        {1, 0, 0},
                        {1, 0, 0},
                        {0, 1, 0},
                        {0, 1, 0},
                        {0, 0, 1},
                        {0, 0, 1},
                        {1, 1, 0},
                        {1, 1, 0},
                        {1, 0, 1},
                        {1, 0, 1},
                        {0, 1, 1},
                        {0, 1, 1}
                },
                groups,
                3,
                42L);

        assertEquals(3, folds.size());
        for (DataLoader.TensorDatasetFold fold : folds) {
            assertEquals(8, fold.train().size());
            assertEquals(4, fold.validation().size());
            assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(fold.validation(), 3));
            for (int group : groupsForRows(fold.validation(), groups)) {
                assertTrue(!groupsForRows(fold.train(), groups).contains(group));
            }
        }
    }

    @Test
    public void testMultiLabelRepeatedStratifiedGroupCrossValidationFacade() {
        int[] groups = {10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60};
        var folds = Aljabr.DL.multiLabelBinaryRepeatedStratifiedGroupCrossValidationFolds(
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f,
                        7f,
                        8f,
                        9f,
                        10f,
                        11f,
                        12f
                }, 12, 1),
                new int[][] {
                        {1, 0, 0},
                        {1, 0, 0},
                        {0, 1, 0},
                        {0, 1, 0},
                        {0, 0, 1},
                        {0, 0, 1},
                        {1, 1, 0},
                        {1, 1, 0},
                        {1, 0, 1},
                        {1, 0, 1},
                        {0, 1, 1},
                        {0, 1, 1}
                },
                groups,
                3,
                2,
                42L);

        assertEquals(6, folds.size());
        assertEquals(6, folds.get(0).foldCount());
        for (DataLoader.TensorDatasetFold fold : folds) {
            assertEquals(8, fold.train().size());
            assertEquals(4, fold.validation().size());
            assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(fold.validation(), 3));
            for (int group : groupsForRows(fold.validation(), groups)) {
                assertTrue(!groupsForRows(fold.train(), groups).contains(group));
            }
        }
    }

    @Test
    public void testStratifiedCrossValidationFacade() {
        var folds = Aljabr.DL.classificationStratifiedCrossValidationFolds(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0.8f, 0.2f,
                        0.7f, 0.3f,
                        0f, 1f,
                        0.1f, 0.9f,
                        0.2f, 0.8f,
                        0.3f, 0.7f
                }, 8, 2),
                new int[] {0, 0, 0, 0, 1, 1, 1, 1},
                4,
                42L);

        assertEquals(4, folds.size());
        assertEquals(6, folds.get(0).train().size());
        assertEquals(2, folds.get(0).validation().size());
        assertEquals(3, folds.get(0).trainLoader(2).numBatches());
        assertEquals(1, folds.get(0).validationLoader(2).numBatches());
    }

    @Test
    public void testGroupCrossValidationFacade() {
        var folds = Aljabr.DL.groupCrossValidationFolds(
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                new int[] {10, 10, 20, 20, 30, 30},
                3,
                42L);

        assertEquals(3, folds.size());
        assertEquals(4, folds.get(0).train().size());
        assertEquals(2, folds.get(0).validation().size());
        assertEquals(3L, folds.stream().mapToInt(fold -> fold.validation().size()).count());
        assertEquals(6, folds.stream().mapToInt(fold -> fold.validation().size()).sum());
    }

    @Test
    public void testRepeatedCrossValidationFacade() {
        var folds = Aljabr.DL.repeatedCrossValidationFolds(
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                3,
                2,
                42L);

        assertEquals(6, folds.size());
        assertEquals(6, folds.get(0).foldCount());
        assertEquals(0, folds.get(0).foldIndex());
        assertEquals(5, folds.get(5).foldIndex());
        assertEquals(12, folds.stream().mapToInt(fold -> fold.validation().size()).sum());
    }

    @Test
    public void testStratifiedGroupCrossValidationFacade() {
        var folds = Aljabr.DL.classificationStratifiedGroupCrossValidationFolds(
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f,
                        7f,
                        8f
                }, 8, 1),
                new int[] {0, 0, 0, 0, 1, 1, 1, 1},
                new int[] {10, 10, 20, 20, 30, 30, 40, 40},
                2,
                42L);

        assertEquals(2, folds.size());
        assertEquals(4, folds.get(0).train().size());
        assertEquals(4, folds.get(0).validation().size());
        assertEquals(8, folds.stream().mapToInt(fold -> fold.validation().size()).sum());
    }

    @Test
    public void testTimeSeriesCrossValidationFacadeWalksForward() {
        var folds = Aljabr.DL.timeSeriesCrossValidationFolds(
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                2);

        assertEquals(2, folds.size());
        assertEquals(2, folds.get(0).train().size());
        assertEquals(2, folds.get(0).validation().size());
        assertEquals(4, folds.get(1).train().size());
        assertEquals(2, folds.get(1).validation().size());
    }

    @Test
    public void testFitCrossValidationFacadeRunsAllFoldsAndAggregatesLosses() {
        var folds = Aljabr.DL.crossValidationFolds(
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                3,
                11L);

        CrossValidationTrainingSummary summary = Aljabr.DL.fitCrossValidation(
                IdentityModel::new,
                folds,
                2,
                false,
                0L,
                1,
                0.01f,
                Aljabr.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Aljabr.DL.trainingOptions()
                        .device("cpu")
                        .saveBestModelCheckpoint(false)
                        .meanAbsoluteErrorMetric()
                        .build());

        assertEquals(3, summary.foldCount());
        assertEquals(3, summary.folds().size());
        assertEquals(3, summary.latestTrainLoss().finiteCount());
        assertEquals(3, summary.latestValidationLoss().finiteCount());
        assertEquals(0.0, summary.meanLatestValidationLoss(), 1e-6);
        assertEquals(3, summary.latestValidationMetrics().get("mae").finiteCount());
        assertEquals(0.0, summary.meanLatestValidationMetric("mae"), 1e-6);
        assertEquals(0.0, summary.meanLatestTrainMetric("mae"), 1e-6);
        assertTrue(summary.bestFoldByValidationMetric("mae", false).isPresent());
        assertTrue(summary.bestFoldByValidationLoss().isPresent());
        assertEquals(3, summary.foldsRankedByValidationLoss().size());
        assertEquals(3, summary.foldsRankedByValidationMetric("mae", false).size());
        assertEquals(3, summary.foldReports().size());
        assertEquals(1, summary.foldReports().get(0).foldNumber());
        assertEquals(0.0, summary.foldReports().get(0).latestValidationMetrics().get("mae"), 1e-6);
        assertEquals(3, summary.foldReportRows().size());
        assertEquals(1, summary.foldReportRows().get(0).get("foldNumber"));
        assertEquals(4, summary.folds().get(0).trainSize());
        assertEquals(2, summary.folds().get(0).validationSize());
    }

    @Test
    public void testFitKFoldFacadeBuildsFoldsAndTrainsInOneCall() {
        CrossValidationTrainingSummary summary = Aljabr.DL.fitKFold(
                IdentityModel::new,
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                3,
                11L,
                2,
                1,
                0.01f,
                Aljabr.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Aljabr.DL.trainingOptions()
                        .device("cpu")
                        .saveBestModelCheckpoint(false)
                        .meanAbsoluteErrorMetric()
                        .build());

        assertEquals(3, summary.foldCount());
        assertEquals(0.0, summary.meanLatestValidationLoss(), 1e-6);
        assertEquals(0.0, summary.meanLatestValidationMetric("mae"), 1e-6);
    }

    @Test
    public void testFitRepeatedKFoldFacadeBuildsFoldsAndTrainsInOneCall() {
        CrossValidationTrainingSummary summary = Aljabr.DL.fitRepeatedKFold(
                IdentityModel::new,
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                3,
                2,
                11L,
                2,
                1,
                0.01f,
                Aljabr.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Aljabr.DL.trainingOptions()
                        .device("cpu")
                        .saveBestModelCheckpoint(false)
                        .meanAbsoluteErrorMetric()
                        .build());

        assertEquals(6, summary.foldCount());
        assertEquals(6, summary.latestValidationLoss().finiteCount());
        assertEquals(0.0, summary.meanLatestValidationLoss(), 1e-6);
        assertEquals(0.0, summary.meanLatestValidationMetric("mae"), 1e-6);
    }

    @Test
    public void testFitGroupKFoldFacadeBuildsFoldsAndTrainsInOneCall() {
        CrossValidationTrainingSummary summary = Aljabr.DL.fitGroupKFold(
                IdentityModel::new,
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                new int[] {10, 10, 20, 20, 30, 30},
                3,
                11L,
                2,
                1,
                0.01f,
                Aljabr.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Aljabr.DL.trainingOptions()
                        .device("cpu")
                        .saveBestModelCheckpoint(false)
                        .meanAbsoluteErrorMetric()
                        .build());

        assertEquals(3, summary.foldCount());
        assertEquals(4, summary.folds().get(0).trainSize());
        assertEquals(2, summary.folds().get(0).validationSize());
        assertEquals(0.0, summary.meanLatestValidationLoss(), 1e-6);
        assertEquals(0.0, summary.meanLatestValidationMetric("mae"), 1e-6);
    }

    @Test
    public void testFitTimeSeriesSplitFacadeTrainsWithoutShufflingTrainingWindows() {
        CrossValidationTrainingSummary summary = Aljabr.DL.fitTimeSeriesSplit(
                IdentityModel::new,
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                GradTensor.of(new float[] {
                        1f,
                        2f,
                        3f,
                        4f,
                        5f,
                        6f
                }, 6, 1),
                2,
                2,
                1,
                0.01f,
                Aljabr.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Aljabr.DL.trainingOptions()
                        .device("cpu")
                        .saveBestModelCheckpoint(false)
                        .meanAbsoluteErrorMetric()
                        .build());

        assertEquals(2, summary.foldCount());
        assertEquals(2, summary.latestValidationLoss().finiteCount());
        assertEquals(false, summary.metadata().get("shuffleTraining"));
        assertEquals(0.0, summary.meanLatestValidationLoss(), 1e-6);
        assertEquals(0.0, summary.meanLatestValidationMetric("mae"), 1e-6);
    }

    @Test
    public void testFitClassificationStratifiedGroupKFoldFacadePreservesGroupsAndTrains() {
        CrossValidationTrainingSummary summary = Aljabr.DL.fitClassificationStratifiedGroupKFold(
                IdentityModel::new,
                GradTensor.of(new float[] {
                        4f, 1f,
                        3f, 1f,
                        2f, 0f,
                        5f, 1f,
                        1f, 4f,
                        1f, 3f,
                        0f, 2f,
                        1f, 5f
                }, 8, 2),
                new int[] {0, 0, 0, 0, 1, 1, 1, 1},
                new int[] {10, 10, 20, 20, 30, 30, 40, 40},
                2,
                42L,
                2,
                1,
                0.01f,
                Aljabr.DL.TrainingPreset.CLASSIFICATION_CROSS_ENTROPY_SGD,
                Aljabr.DL.trainingOptions()
                        .device("cpu")
                        .saveBestModelCheckpoint(false)
                        .accuracyMetric()
                        .build());

        assertEquals(2, summary.foldCount());
        assertEquals(2, summary.latestValidationLoss().finiteCount());
        assertEquals(1.0, summary.meanLatestValidationMetric("accuracy"), 1e-6);
    }

    @Test
    public void testFitClassificationStratifiedKFoldFacadePreservesLabelsAndTrains() {
        CrossValidationTrainingSummary summary = Aljabr.DL.fitClassificationStratifiedKFold(
                IdentityModel::new,
                GradTensor.of(new float[] {
                        4f, 1f,
                        3f, 1f,
                        2f, 0f,
                        5f, 1f,
                        1f, 4f,
                        1f, 3f,
                        0f, 2f,
                        1f, 5f
                }, 8, 2),
                new int[] {0, 0, 0, 0, 1, 1, 1, 1},
                4,
                42L,
                2,
                1,
                0.01f,
                Aljabr.DL.TrainingPreset.CLASSIFICATION_CROSS_ENTROPY_SGD,
                Aljabr.DL.trainingOptions()
                        .device("cpu")
                        .saveBestModelCheckpoint(false)
                        .accuracyMetric()
                        .build());

        assertEquals(4, summary.foldCount());
        assertEquals(4, summary.latestValidationLoss().finiteCount());
        assertEquals(4, summary.latestValidationMetrics().get("accuracy").finiteCount());
        assertEquals(1.0, summary.meanLatestValidationMetric("accuracy"), 1e-6);
        assertTrue(summary.bestFoldByValidationMetric("accuracy", true).isPresent());
    }

    private static final class IdentityModel extends NNModule {
        @Override
        public GradTensor forward(GradTensor input) {
            return input;
        }
    }

    private static int[] positiveCounts(DataLoader.TensorDataset dataset, int columns) {
        int[] counts = new int[columns];
        for (int i = 0; i < dataset.size(); i++) {
            float[] labels = dataset.get(i)[1].data();
            assertEquals(columns, labels.length);
            for (int column = 0; column < columns; column++) {
                if (labels[column] >= 0.5f) {
                    counts[column]++;
                }
            }
        }
        return counts;
    }

    private static int[] positiveCounts(DataLoader.Batch batch, int columns) {
        int[] counts = new int[columns];
        float[] labels = batch.labels().data();
        assertEquals(0, labels.length % columns);
        for (int row = 0; row < labels.length / columns; row++) {
            for (int column = 0; column < columns; column++) {
                if (labels[row * columns + column] >= 0.5f) {
                    counts[column]++;
                }
            }
        }
        return counts;
    }

    private static java.util.List<Float> flattenInputs(DataLoader.TensorDataLoader loader) {
        java.util.List<Float> values = new java.util.ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            for (float value : batch.inputs().data()) {
                values.add(value);
            }
        }
        return values;
    }

    private static java.util.Set<Integer> groupsForRows(DataLoader.TensorDataset dataset, int[] groups) {
        java.util.Set<Integer> result = new java.util.HashSet<>();
        for (int i = 0; i < dataset.size(); i++) {
            result.add(groups[(int) dataset.get(i)[0].data()[0] - 1]);
        }
        return result;
    }

    private static Map<String, Object> runtimeProfileReport(long forwardNanos, long lossNanos) {
        double forwardMillis = forwardNanos / 1_000_000.0;
        double lossMillis = lossNanos / 1_000_000.0;
        double totalMillis = forwardMillis + lossMillis;
        Map<String, Object> trainGroup = Map.of(
                "name", "train",
                "count", 2L,
                "totalMillis", totalMillis,
                "percentTotal", 100.0,
                "averageMillis", totalMillis / 2.0,
                "minMillis", Math.min(forwardMillis, lossMillis),
                "maxMillis", Math.max(forwardMillis, lossMillis),
                "lastMillis", lossMillis,
                "stddevMillis", Math.abs(lossMillis - forwardMillis) / 2.0);
        Map<String, Object> lossHotspot = Map.of(
                "phase", "train.loss",
                "count", 1L,
                "totalMillis", lossMillis,
                "percentTotal", totalMillis == 0.0 ? 0.0 : lossMillis * 100.0 / totalMillis,
                "averageMillis", lossMillis,
                "minMillis", lossMillis,
                "maxMillis", lossMillis,
                "lastMillis", lossMillis,
                "stddevMillis", 0.0);
        Map<String, Object> metadata = Map.ofEntries(
                Map.entry("runtimeProfile.groupCount", 1),
                Map.entry("runtimeProfile.groups", List.of(trainGroup)),
                Map.entry("runtimeProfile.primaryGroup.name", "train"),
                Map.entry("runtimeProfile.primaryGroup.totalMillis", totalMillis),
                Map.entry("runtimeProfile.primaryGroup.percentTotal", 100.0),
                Map.entry("runtimeProfile.primaryGroup.averageMillis", totalMillis / 2.0),
                Map.entry("runtimeProfile.primaryGroup.minMillis", Math.min(forwardMillis, lossMillis)),
                Map.entry("runtimeProfile.primaryGroup.lastMillis", lossMillis),
                Map.entry("runtimeProfile.primaryGroup.stddevMillis", Math.abs(lossMillis - forwardMillis) / 2.0),
                Map.entry("runtimeProfile.hotspotCount", 2),
                Map.entry("runtimeProfile.hotspots", List.of(
                        Map.of(
                                "phase", "train.forward",
                                "count", 1L,
                                "totalMillis", forwardMillis,
                                "percentTotal", totalMillis == 0.0 ? 0.0 : forwardMillis * 100.0 / totalMillis,
                                "averageMillis", forwardMillis,
                                "minMillis", forwardMillis,
                                "maxMillis", forwardMillis,
                                "lastMillis", forwardMillis,
                                "stddevMillis", 0.0),
                        lossHotspot)),
                Map.entry("runtimeProfile.primaryHotspot.phase", "train.loss"),
                Map.entry("runtimeProfile.primaryHotspot.totalMillis", lossMillis),
                Map.entry("runtimeProfile.primaryHotspot.percentTotal", totalMillis == 0.0 ? 0.0 : lossMillis * 100.0 / totalMillis),
                Map.entry("runtimeProfile.primaryHotspot.averageMillis", lossMillis),
                Map.entry("runtimeProfile.primaryHotspot.minMillis", lossMillis),
                Map.entry("runtimeProfile.primaryHotspot.lastMillis", lossMillis),
                Map.entry("runtimeProfile.primaryHotspot.stddevMillis", 0.0));
        return Map.of(
                "diagnostics", List.of(),
                "metadata", metadata);
    }

    private static int[][] balancedGroupedMultiLabelRows() {
        return new int[][] {
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1},
                {0, 1, 0}
        };
    }
}
