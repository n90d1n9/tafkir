package tech.kayys.tafkir.ml.data;

import java.util.List;
import tech.kayys.tafkir.train.data.internal.DataLoaderTensorCollationRules;

final class TensorBatchCollator {
    private TensorBatchCollator() {
    }

    static DataLoader.Batch collate(
            DataLoader.CollateFn collateFn,
            List<Integer> batchIndices,
            DataLoader.TensorDatasetAdapter dataset) {
        return DataLoaderTensorCollationRules.collate(collateFn::collate, batchIndices, dataset);
    }
}
