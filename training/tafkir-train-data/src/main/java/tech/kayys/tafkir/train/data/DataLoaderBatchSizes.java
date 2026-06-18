package tech.kayys.tafkir.train.data;

import tech.kayys.tafkir.train.data.internal.DataLoaderBatchSizeRules;

final class DataLoaderBatchSizes {
    private DataLoaderBatchSizes() {
    }

    static int requirePositive(int batchSize) {
        return DataLoaderBatchSizeRules.requirePositive(batchSize);
    }
}
