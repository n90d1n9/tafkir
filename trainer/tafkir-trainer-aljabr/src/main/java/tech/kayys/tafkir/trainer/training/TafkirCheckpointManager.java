package tech.kayys.tafkir.trainer.training;

import java.io.*;
import java.nio.file.*;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Checkpoint manager for Tafkir training.
 *
 * <p>Saves and loads training checkpoints containing model and optimizer state.
 * Supports async saving, compression, and keeping only the top-K best checkpoints.
 */
public final class TafkirCheckpointManager implements AutoCloseable {

    private final Path checkpointDir;
    private final int maxCheckpoints;
    private final boolean compress;
    private final boolean asyncSave;

    private final PriorityQueue<CheckpointMeta> checkpointQueue;
    private final ExecutorService executor;

    /**
     * Creates a new checkpoint manager.
     *
     * @param checkpointDir  directory to save checkpoints
     * @param maxCheckpoints maximum number of checkpoints to keep (older/worse ones are deleted)
     * @param compress       whether to compress checkpoints with GZIP
     * @param asyncSave      whether to perform I/O asynchronously to avoid blocking training
     */
    public TafkirCheckpointManager(String checkpointDir, int maxCheckpoints, boolean compress, boolean asyncSave) {
        this.checkpointDir = Paths.get(checkpointDir);
        this.maxCheckpoints = maxCheckpoints;
        this.compress = compress;
        this.asyncSave = asyncSave;
        
        // Keep checkpoints sorted by validation loss (lowest first)
        // Note: Java PriorityQueue puts smallest at the head. To keep top-K BEST (lowest loss),
        // we want a MAX-heap of the lowest losses, so we evict the highest loss.
        this.checkpointQueue = new PriorityQueue<>(Comparator.comparingDouble(CheckpointMeta::validationLoss).reversed());
        
        this.executor = asyncSave 
            ? Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "tafkir-checkpoint-saver");
                t.setDaemon(true);
                return t;
            })
            : null;

        try {
            Files.createDirectories(this.checkpointDir);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create checkpoint directory: " + checkpointDir, e);
        }
    }

    /**
     * Saves a checkpoint.
     *
     * @param epoch          current epoch
     * @param trainLoss      training loss
     * @param valLoss        validation loss (used for ranking checkpoints)
     * @param modelState     serialized model state
     * @param optimizerState serialized optimizer state
     */
    public void save(int epoch, double trainLoss, double valLoss, byte[] modelState, byte[] optimizerState) {
        String timestamp = DateTimeFormatter.ISO_INSTANT.format(Instant.now());
        String filename = String.format("ckpt_ep%d_loss%.4f_%s.bin%s", 
                epoch, valLoss, timestamp.replace(":", ""), compress ? ".gz" : "");
        
        Path path = checkpointDir.resolve(filename);
        CheckpointMeta meta = new CheckpointMeta(epoch, valLoss, trainLoss, timestamp, path.toString());

        Runnable saveTask = () -> {
            try {
                writeCheckpoint(path, meta, modelState, optimizerState);
                synchronized (checkpointQueue) {
                    checkpointQueue.offer(meta);
                    cleanupOldCheckpoints();
                }
            } catch (IOException e) {
                System.err.println("Failed to save checkpoint: " + path);
                e.printStackTrace();
            }
        };

        if (asyncSave && executor != null) {
            executor.submit(saveTask);
        } else {
            saveTask.run();
        }
    }

    private void writeCheckpoint(Path path, CheckpointMeta meta, byte[] modelState, byte[] optimizerState) throws IOException {
        OutputStream out = Files.newOutputStream(path);
        if (compress) {
            out = new GZIPOutputStream(out);
        }

        try (DataOutputStream dos = new DataOutputStream(out)) {
            dos.writeInt(meta.epoch());
            dos.writeDouble(meta.validationLoss());
            dos.writeDouble(meta.trainLoss());
            dos.writeUTF(meta.timestamp());
            
            // Allow null optimizer state
            int optLen = optimizerState != null ? optimizerState.length : -1;
            dos.writeInt(optLen);
            if (optLen > 0) dos.write(optimizerState);
            
            dos.writeInt(modelState.length);
            dos.write(modelState);
        }
    }

    private void cleanupOldCheckpoints() {
        while (checkpointQueue.size() > maxCheckpoints) {
            // PriorityQueue head is the MAX element (highest loss).
            // We remove the highest loss checkpoint and delete its file.
            CheckpointMeta evict = checkpointQueue.poll();
            if (evict != null) {
                try {
                    Files.deleteIfExists(Paths.get(evict.path()));
                } catch (IOException e) {
                    System.err.println("Failed to delete evicted checkpoint: " + evict.path());
                }
            }
        }
    }

    /**
     * Loads the best checkpoint according to validation loss.
     *
     * @return checkpoint data, or null if no checkpoints exist
     */
    public CheckpointData loadBest() throws IOException {
        CheckpointMeta best = null;
        synchronized (checkpointQueue) {
            if (checkpointQueue.isEmpty()) return null;
            // Iterate to find the MIN element (lowest loss)
            for (CheckpointMeta m : checkpointQueue) {
                if (best == null || m.validationLoss() < best.validationLoss()) {
                    best = m;
                }
            }
        }
        if (best == null) return null;
        return load(best.path());
    }

    /**
     * Loads a checkpoint from a specific path.
     */
    public CheckpointData load(String path) throws IOException {
        InputStream in = Files.newInputStream(Paths.get(path));
        if (path.endsWith(".gz")) {
            in = new GZIPInputStream(in);
        }

        try (DataInputStream dis = new DataInputStream(in)) {
            int epoch = dis.readInt();
            double valLoss = dis.readDouble();
            double trainLoss = dis.readDouble();
            String timestamp = dis.readUTF();
            
            int optLen = dis.readInt();
            byte[] optimizerState = null;
            if (optLen > 0) {
                optimizerState = new byte[optLen];
                dis.readFully(optimizerState);
            }
            
            int modelLen = dis.readInt();
            byte[] modelState = new byte[modelLen];
            dis.readFully(modelState);
            
            return new CheckpointData(epoch, trainLoss, valLoss, timestamp, modelState, optimizerState);
        }
    }

    /** Returns metadata for all currently managed checkpoints. */
    public List<CheckpointMeta> listCheckpoints() {
        synchronized (checkpointQueue) {
            return new ArrayList<>(checkpointQueue);
        }
    }

    @Override
    public void close() {
        if (executor != null) {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    public record CheckpointMeta(int epoch, double validationLoss, double trainLoss, String timestamp, String path) {}
    public record CheckpointData(int epoch, double trainLoss, double validationLoss, String timestamp, byte[] modelState, byte[] optimizerState) {}
}
