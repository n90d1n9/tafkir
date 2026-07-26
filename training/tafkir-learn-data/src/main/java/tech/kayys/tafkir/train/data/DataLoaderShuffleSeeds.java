package tech.kayys.tafkir.train.data;

import tech.kayys.tafkir.train.data.internal.DataLoaderShuffleSeedRules;

final class DataLoaderShuffleSeeds {
    private DataLoaderShuffleSeeds() {
    }

    static Long forEpoch(Long baseSeed, boolean reshuffleEachEpoch, long epoch) {
        return DataLoaderShuffleSeedRules.forEpoch(baseSeed, reshuffleEachEpoch, epoch);
    }
}
