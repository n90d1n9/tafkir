package tech.kayys.tafkir.distributed;

import tech.kayys.tafkir.ml.tensor.TafkirTensor;
import tech.kayys.tafkir.trainer.model.TafkirModel;
import tech.kayys.tafkir.trainer.loss.TafkirLoss;
import tech.kayys.tafkir.trainer.optim.TafkirOptimizer;

import java.util.List;
import java.util.concurrent.*;

/**
 * Distributed data parallel training across multiple threads.
 */
public final class DistributedTrainer {

    private final int numWorkers;
    private final ExecutorService executor;

    public DistributedTrainer(int numWorkers) {
        this.numWorkers = numWorkers;
        this.executor = Executors.newFixedThreadPool(numWorkers);
    }

    public void fit(TafkirModel model, TafkirLoss lossFn, TafkirOptimizer optimizer,
                     TafkirTensor x, TafkirTensor y, int epochs, int batchSize) {

        int numSamples = (int) x.shape()[0];
        int workerBatchSize = batchSize / numWorkers;
        model.train();
        List<TafkirTensor> mainParams = model.parameters();

        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0;
            int numBatches = numSamples / batchSize;

            for (int batch = 0; batch < numBatches; batch++) {
                int batchStart = batch * batchSize;
                List<Future<WorkerResult>> futures = new java.util.ArrayList<>();

                for (int w = 0; w < numWorkers; w++) {
                    final int start = batchStart + w * workerBatchSize;
                    final int end = start + workerBatchSize;
                    futures.add(executor.submit(() -> {
                        TafkirTensor wx = x.slice(new long[]{start, 0}, new long[]{end - start, x.shape()[1]});
                        TafkirTensor wy = y.slice(new long[]{start}, new long[]{end - start});
                        TafkirTensor pred = model.forward(wx);
                        TafkirTensor loss = lossFn.compute(pred, wy);
                        loss.backward();
                        List<TafkirTensor> grads = new java.util.ArrayList<>();
                        for (TafkirTensor p : model.parameters()) grads.add(p.gradTensor());
                        return new WorkerResult(loss.item(), grads);
                    }));
                }

                float batchLoss = 0;
                List<List<TafkirTensor>> allGrads = new java.util.ArrayList<>();
                for (Future<WorkerResult> f : futures) {
                    try {
                        WorkerResult r = f.get();
                        batchLoss += r.loss;
                        allGrads.add(r.gradients);
                    } catch (InterruptedException | ExecutionException e) {
                        throw new RuntimeException("Worker failed", e);
                    }
                }

                for (int p = 0; p < mainParams.size(); p++) {
                    TafkirTensor avgGrad = allGrads.get(0).get(p);
                    for (int w = 1; w < numWorkers; w++) avgGrad.add_(allGrads.get(w).get(p));
                    avgGrad.div_(numWorkers);
                    mainParams.get(p).setGrad(avgGrad.unwrap());
                }

                optimizer.step(mainParams);
                optimizer.zeroGrad(mainParams);
                epochLoss += batchLoss / numWorkers;
            }
            System.out.printf("Epoch %d/%d, avg loss: %.4f%n", epoch + 1, epochs, epochLoss / numBatches);
        }
    }

    public void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) executor.shutdownNow();
        } catch (InterruptedException e) { executor.shutdownNow(); }
    }

    private record WorkerResult(float loss, List<TafkirTensor> gradients) {}
}
