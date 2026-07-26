package tech.kayys.tafkir.learn;

import tech.kayys.tafkir.nn.NNModule;
import tech.kayys.tafkir.nn.Parameter;
import tech.kayys.tafkir.nn.layer.*;
import tech.kayys.tafkir.nn.loss.*;
import tech.kayys.tafkir.nn.transformer.*;
import tech.kayys.tafkir.nn.optimize.*;
import tech.kayys.tafkir.train.data.Dataset;
import tech.kayys.tafkir.train.data.DataLoader;
import tech.kayys.tafkir.ml.train.CanonicalTrainer;
import tech.kayys.tafkir.ml.train.TrainingConfig;

/**
 * Tafkir-Learn: A unified deep learning framework for the JVM.
 * 
 * This is the main entry point for Tafkir-Learn, providing PyTorch-style imports
 * and a fluent API for building, training, and deploying neural networks.
 * 
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * import static tech.kayys.tafkir.learn.TafkirLearn.*;
 * 
 * // Define a model
 * var model = nn.Sequential(
 *     new Linear(784, 512),
 *     new ReLU(),
 *     new Dropout(0.5),
 *     new Linear(512, 10)
 * );
 * 
 * // Create optimizer
 * var optimizer = optim.Adam(model.parameters(), 0.001);
 * 
 * // Setup data
 * var dataset = new MyDataset();
 * var loader = data.DataLoader(dataset, 32);
 * 
 * // Train
 * var config = TrainingConfig.builder()
 *     .epochs(10)
 *     .learningRate(0.001)
 *     .build();
 * var trainer = trainer.create(model, optimizer, config);
 * trainer.train(loader);
 * }</pre>
 * 
 * @author Tafkir Team
 * @version 0.1.0
 */
public final class TafkirLearn {
    
    /**
     * Neural network modules and layers namespace.
     * Equivalent to {@code torch.nn} in PyTorch.
     */
    public static final NN nn = new NN();
    
    /**
     * Optimization algorithms namespace.
     * Equivalent to {@code torch.optim} in PyTorch.
     */
    public static final Optim optim = new Optim();
    
    /**
     * Data loading and preprocessing namespace.
     * Equivalent to {@code torch.utils.data} in PyTorch.
     */
    public static final Data data = new Data();
    
    /**
     * Training utilities namespace.
     */
    public static final TrainerAPI trainer = new TrainerAPI();
    
    private TafkirLearn() {
        // Prevent instantiation
    }
    
    /**
     * Neural network module factory.
     */
    public static final class NN {
        
        public Sequential Sequential(NNModule... modules) {
            return new Sequential(modules);
        }
        
        public Linear Linear(int inFeatures, int outFeatures) {
            return new Linear(inFeatures, outFeatures);
        }
        
        public Linear Linear(int inFeatures, int outFeatures, boolean bias) {
            return new Linear(inFeatures, outFeatures, bias);
        }
        
        public Conv2d Conv2d(int inChannels, int outChannels, int kernelSize) {
            return new Conv2d(inChannels, outChannels, kernelSize);
        }
        
        public ReLU ReLU() {
            return new ReLU();
        }
        
        public GELU GELU() {
            return new GELU();
        }
        
        public SiLU SiLU() {
            return new SiLU();
        }
        
        public ELU ELU() {
            return new ELU();
        }
        
        public LeakyReLU LeakyReLU() {
            return new LeakyReLU();
        }
        
        public LeakyReLU LeakyReLU(double negativeSlope) {
            return new LeakyReLU(negativeSlope);
        }
        
        public Dropout Dropout(double p) {
            return new Dropout(p);
        }
        
        public LayerNorm LayerNorm(int normalizedShape) {
            return new LayerNorm(normalizedShape);
        }
        
        public GroupNorm GroupNorm(int numGroups, int numChannels) {
            return new GroupNorm(numGroups, numChannels);
        }
        
        public Embedding Embedding(int numEmbeddings, int embeddingDim) {
            return new Embedding(numEmbeddings, embeddingDim);
        }
        
        public Flatten Flatten() {
            return new Flatten();
        }
        
        public ResidualBlock ResidualBlock(NNModule... layers) {
            return new ResidualBlock(layers);
        }
        
        // Loss functions
        public MSELoss MSELoss() {
            return new MSELoss();
        }
        
        public CrossEntropyLoss CrossEntropyLoss() {
            return new CrossEntropyLoss();
        }
        
        public BCEWithLogitsLoss BCEWithLogitsLoss() {
            return new BCEWithLogitsLoss();
        }
        
        public L1Loss L1Loss() {
            return new L1Loss();
        }
        
        public SmoothL1Loss SmoothL1Loss() {
            return new SmoothL1Loss();
        }
        
        public HuberLoss HuberLoss(double delta) {
            return new HuberLoss(delta);
        }
        
        // Transformer components
        public MultiHeadAttention MultiHeadAttention(int embedDim, int numHeads) {
            return new MultiHeadAttention(embedDim, numHeads);
        }
        
        public TransformerEncoderLayer TransformerEncoderLayer(int dModel, int nhead) {
            return new TransformerEncoderLayer(dModel, nhead);
        }
        
        public TransformerEncoder TransformerEncoder(TransformerEncoderLayer[] layers, int numLayers) {
            return new TransformerEncoder(layers, numLayers);
        }
        
        public PositionalEncoding PositionalEncoding(int dModel, int maxLen) {
            return new PositionalEncoding(dModel, maxLen);
        }
    }
    
    /**
     * Optimizer factory.
     */
    public static final class Optim {
        
        public SGD SGD(java.util.List<Parameter> parameters, double lr) {
            return new SGD(parameters, lr);
        }
        
        public SGD SGD(java.util.List<Parameter> parameters, double lr, double momentum) {
            return new SGD(parameters, lr, momentum);
        }
        
        public Adam Adam(java.util.List<Parameter> parameters, double lr) {
            return new Adam(parameters, lr);
        }
        
        public AdamW AdamW(java.util.List<Parameter> parameters, double lr) {
            return new AdamW(parameters, lr);
        }
        
        public RMSprop RMSprop(java.util.List<Parameter> parameters, double lr) {
            return new RMSprop(parameters, lr);
        }
        
        public Adagrad Adagrad(java.util.List<Parameter> parameters, double lr) {
            return new Adagrad(parameters, lr);
        }
        
        public Lion Lion(java.util.List<Parameter> parameters, double lr) {
            return new Lion(parameters, lr);
        }
        
        public LAMB LAMB(java.util.List<Parameter> parameters, double lr) {
            return new LAMB(parameters, lr);
        }
        
        public Lookahead Lookahead(Optimizer baseOptimizer) {
            return new Lookahead(baseOptimizer);
        }
        
        public SAM SAM(Optimizer baseOptimizer) {
            return new SAM(baseOptimizer);
        }
    }
    
    /**
     * Data loading utilities.
     */
    public static final class Data {
        
        public <T> DataLoader<T> DataLoader(Dataset<T> dataset, int batchSize) {
            return new DataLoader<>(dataset, batchSize);
        }
        
        public <T> DataLoader<T> DataLoader(Dataset<T> dataset, int batchSize, boolean shuffle) {
            return new DataLoader<>(dataset, batchSize, shuffle);
        }
    }
    
    /**
     * Training API utilities.
     */
    public static final class TrainerAPI {
        
        public CanonicalTrainer create(NNModule model, Optimizer optimizer, TrainingConfig config) {
            return CanonicalTrainer.create(model, optimizer, config);
        }
    }
}
