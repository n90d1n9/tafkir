package tech.kayys.tafkir.nn.functional;

import tech.kayys.tafkir.ml.autograd.GradTensor;
import tech.kayys.tafkir.ml.nn.Parameter;

/**
 * Functional interface for neural network operations.
 * <p>
 * Provides stateless functions that can be used without creating layer instances,
 * similar to {@code torch.nn.functional} in PyTorch.
 * <p>
 * These functions are useful when you need more flexibility than module-based APIs,
 * such as custom architectures or weight-sharing scenarios.
 * 
 * <h3>Example Usage</h3>
 * <pre>{@code
 * import static tech.kayys.tafkir.nn.functional.F.*;
 * 
 * // Functional linear transformation
 * var output = linear(input, weight, bias);
 * 
 * // Functional activation
 * var activated = relu(output);
 * 
 * // Functional loss
 * var loss = cross_entropy(logits, targets);
 * }</pre>
 */
public final class F {
    
    private F() {
        // Prevent instantiation
    }
    
    /**
     * Applies a linear transformation: {@code y = x @ W^T + b}.
     * <p>
     * This is the functional version of {@link tech.kayys.tafkir.nn.layer.Linear}.
     * 
     * <h3>Shape</h3>
     * <ul>
     *   <li><b>Input:</b> {@code [..., inFeatures]}</li>
     *   <li><b>Weight:</b> {@code [outFeatures, inFeatures]}</li>
     *   <li><b>Bias:</b> {@code [1, outFeatures]} (optional, can be null)</li>
     *   <li><b>Output:</b> {@code [..., outFeatures]}</li>
     * </ul>
     * 
     * @param input input tensor of shape {@code [..., inFeatures]}
     * @param weight weight matrix of shape {@code [outFeatures, inFeatures]}
     * @param bias bias vector of shape {@code [1, outFeatures]} or null
     * @return transformed tensor of shape {@code [..., outFeatures]}
     */
    public static GradTensor linear(GradTensor input, GradTensor weight, GradTensor bias) {
        GradTensor result = input.matmul(weight.transpose());
        if (bias != null) {
            result = result.add(bias);
        }
        return result;
    }
    
    /**
     * Applies a linear transformation without bias: {@code y = x @ W^T}.
     * 
     * @param input input tensor of shape {@code [..., inFeatures]}
     * @param weight weight matrix of shape {@code [outFeatures, inFeatures]}
     * @return transformed tensor of shape {@code [..., outFeatures]}
     */
    public static GradTensor linear(GradTensor input, GradTensor weight) {
        return linear(input, weight, null);
    }
    
    /**
     * Applies the Rectified Linear Unit (ReLU) function element-wise.
     * <p>
     * {@code ReLU(x) = max(0, x)}
     * 
     * <h3>Shape</h3>
     * <ul>
     *   <li><b>Input:</b> Arbitrary shape</li>
     *   <li><b>Output:</b> Same shape as input</li>
     * </ul>
     * 
     * @param input input tensor
     * @return ReLU-activated tensor
     */
    public static GradTensor relu(GradTensor input) {
        return input.relu();
    }
    
    /**
     * Applies the GELU (Gaussian Error Linear Unit) function element-wise.
     * <p>
     * {@code GELU(x) = x * Φ(x)} where Φ is the standard Gaussian CDF.
     * 
     * @param input input tensor
     * @return GELU-activated tensor
     */
    public static GradTensor gelu(GradTensor input) {
        return input.gelu();
    }
    
    /**
     * Applies the Sigmoid Linear Unit (SiLU/Swish) function element-wise.
     * <p>
     * {@code SiLU(x) = x * sigmoid(x)}
     * 
     * @param input input tensor
     * @return SiLU-activated tensor
     */
    public static GradTensor silu(GradTensor input) {
        return input.silu();
    }
    
    /**
     * Applies the Exponential Linear Unit (ELU) function element-wise.
     * <p>
     * {@code ELU(x) = x if x > 0 else α * (exp(x) - 1)}
     * 
     * @param input input tensor
     * @param alpha scaling factor for negative values (default: 1.0)
     * @return ELU-activated tensor
     */
    public static GradTensor elu(GradTensor input, double alpha) {
        return input.elu(alpha);
    }
    
    /**
     * Applies the Exponential Linear Unit (ELU) function with alpha=1.0.
     * 
     * @param input input tensor
     * @return ELU-activated tensor
     */
    public static GradTensor elu(GradTensor input) {
        return elu(input, 1.0);
    }
    
    /**
     * Applies the Leaky Rectified Linear Unit (LeakyReLU) function element-wise.
     * <p>
     * {@code LeakyReLU(x) = x if x > 0 else negativeSlope * x}
     * 
     * @param input input tensor
     * @param negativeSlope slope for negative values (default: 0.01)
     * @return LeakyReLU-activated tensor
     */
    public static GradTensor leakyRelu(GradTensor input, double negativeSlope) {
        return input.leakyRelu(negativeSlope);
    }
    
    /**
     * Applies the LeakyReLU function with negativeSlope=0.01.
     * 
     * @param input input tensor
     * @return LeakyReLU-activated tensor
     */
    public static GradTensor leakyRelu(GradTensor input) {
        return leakyRelu(input, 0.01);
    }
    
    /**
     * Applies dropout to the input tensor during training.
     * <p>
     * Randomly zeros elements with probability p and scales remaining elements by 1/(1-p).
     * 
     * <h3>Shape</h3>
     * <ul>
     *   <li><b>Input:</b> Arbitrary shape</li>
     *   <li><b>Output:</b> Same shape as input</li>
     * </ul>
     * 
     * @param input input tensor
     * @param p probability of zeroing an element (e.g., 0.5 for 50% dropout)
     * @param training if true, apply dropout; if false, return input unchanged
     * @return tensor with dropout applied
     */
    public static GradTensor dropout(GradTensor input, double p, boolean training) {
        if (!training || p == 0.0) {
            return input;
        }
        return input.dropout(p);
    }
    
    /**
     * Applies dropout with p=0.5 during training.
     * 
     * @param input input tensor
     * @param training if true, apply dropout
     * @return tensor with dropout applied
     */
    public static GradTensor dropout(GradTensor input, boolean training) {
        return dropout(input, 0.5, training);
    }
    
    /**
     * Applies Layer Normalization over a mini-batch of inputs.
     * 
     * <h3>Shape</h3>
     * <ul>
     *   <li><b>Input:</b> {@code [..., normalizedShape]}</li>
     *   <li><b>Weight:</b> {@code [normalizedShape]} (optional)</li>
     *   <li><b>Bias:</b> {@code [normalizedShape]} (optional)</li>
     *   <li><b>Output:</b> Same shape as input</li>
     * </ul>
     * 
     * @param input input tensor
     * @param normalizedShape shape of the dimensions to normalize over
     * @param weight scale parameter (optional, can be null)
     * @param bias shift parameter (optional, can be null)
     * @param eps epsilon value for numerical stability (default: 1e-5)
     * @return normalized tensor
     */
    public static GradTensor layerNorm(GradTensor input, int[] normalizedShape, 
                                       GradTensor weight, GradTensor bias, double eps) {
        return input.layerNorm(normalizedShape, weight, bias, eps);
    }
    
    /**
     * Applies Layer Normalization with default epsilon.
     * 
     * @param input input tensor
     * @param normalizedShape shape of the dimensions to normalize over
     * @param weight scale parameter (optional)
     * @param bias shift parameter (optional)
     * @return normalized tensor
     */
    public static GradTensor layerNorm(GradTensor input, int[] normalizedShape, 
                                       GradTensor weight, GradTensor bias) {
        return layerNorm(input, normalizedShape, weight, bias, 1e-5);
    }
    
    /**
     * Applies Group Normalization over a mini-batch of inputs.
     * 
     * @param input input tensor of shape {@code [N, C, ...]}
     * @param numGroups number of groups to divide channels into
     * @param numChannels total number of channels
     * @param weight scale parameter (optional)
     * @param bias shift parameter (optional)
     * @param eps epsilon value for numerical stability
     * @return normalized tensor
     */
    public static GradTensor groupNorm(GradTensor input, int numGroups, int numChannels,
                                       GradTensor weight, GradTensor bias, double eps) {
        return input.groupNorm(numGroups, numChannels, weight, bias, eps);
    }
    
    /**
     * Applies softmax along the specified dimension.
     * 
     * <h3>Shape</h3>
     * <ul>
     *   <li><b>Input:</b> Arbitrary shape</li>
     *   <li><b>Output:</b> Same shape as input, values sum to 1 along dim</li>
     * </ul>
     * 
     * @param input input tensor
     * @param dim dimension along which to apply softmax
     * @return softmax-normalized tensor
     */
    public static GradTensor softmax(GradTensor input, int dim) {
        return input.softmax(dim);
    }
    
    /**
     * Applies log-softmax along the specified dimension.
     * 
     * @param input input tensor
     * @param dim dimension along which to apply log-softmax
     * @return log-softmax normalized tensor
     */
    public static GradTensor logSoftmax(GradTensor input, int dim) {
        return input.logSoftmax(dim);
    }
    
    /**
     * Applies the sigmoid function element-wise.
     * <p>
     * {@code sigmoid(x) = 1 / (1 + exp(-x))}
     * 
     * @param input input tensor
     * @return sigmoid-activated tensor
     */
    public static GradTensor sigmoid(GradTensor input) {
        return input.sigmoid();
    }
    
    /**
     * Computes the Cross Entropy loss between input logits and target labels.
     * <p>
     * This combines log-softmax and NLLLoss in a single operation for numerical stability.
     * 
     * <h3>Shape</h3>
     * <ul>
     *   <li><b>Input:</b> {@code [batchSize, numClasses]} or {@code [batchSize, numClasses, ...]}</li>
     *   <li><b>Target:</b> {@code [batchSize]} with class indices in range [0, numClasses-1]</li>
     *   <li><b>Output:</b> Scalar loss value</li>
     * </ul>
     * 
     * @param input logits tensor (unnormalized scores)
     * @param target class indices tensor
     * @param weight optional per-class weights (can be null)
     * @param ignoreIndex target value to ignore in loss computation (default: -100)
     * @param reduction reduction method: "mean", "sum", or "none"
     * @return computed loss
     */
    public static GradTensor crossEntropy(GradTensor input, GradTensor target, 
                                          GradTensor weight, int ignoreIndex, String reduction) {
        return tech.kayys.tafkir.nn.loss.CrossEntropyLoss.builder()
            .weight(weight)
            .ignoreIndex(ignoreIndex)
            .reduction(reduction)
            .build()
            .forward(input, target);
    }
    
    /**
     * Computes Cross Entropy loss with default parameters.
     * 
     * @param input logits tensor
     * @param target class indices tensor
     * @return scalar loss (mean reduction)
     */
    public static GradTensor crossEntropy(GradTensor input, GradTensor target) {
        return crossEntropy(input, target, null, -100, "mean");
    }
    
    /**
     * Computes Binary Cross Entropy with Logits loss.
     * <p>
     * This combines sigmoid and BCELoss for numerical stability.
     * 
     * <h3>Shape</h3>
     * <ul>
     *   <li><b>Input:</b> Arbitrary shape</li>
     *   <li><b>Target:</b> Same shape as input, values in [0, 1]</li>
     *   <li><b>Output:</b> Same shape as input (or scalar with reduction)</li>
     * </ul>
     * 
     * @param input logits tensor (unnormalized scores)
     * @param target binary targets tensor
     * @param weight optional per-element weights (can be null)
     * @param posWeight optional weighting factor for positive examples (can be null)
     * @param reduction reduction method: "mean", "sum", or "none"
     * @return computed loss
     */
    public static GradTensor bceWithLogits(GradTensor input, GradTensor target,
                                           GradTensor weight, GradTensor posWeight, String reduction) {
        return tech.kayys.tafkir.nn.loss.BCEWithLogitsLoss.builder()
            .weight(weight)
            .posWeight(posWeight)
            .reduction(reduction)
            .build()
            .forward(input, target);
    }
    
    /**
     * Computes BCEWithLogits loss with default parameters.
     * 
     * @param input logits tensor
     * @param target binary targets tensor
     * @return computed loss (mean reduction)
     */
    public static GradTensor bceWithLogits(GradTensor input, GradTensor target) {
        return bceWithLogits(input, target, null, null, "mean");
    }
    
    /**
     * Computes Mean Squared Error loss.
     * 
     * <h3>Shape</h3>
     * <ul>
     *   <li><b>Input:</b> Arbitrary shape</li>
     *   <li><b>Target:</b> Same shape as input</li>
     *   <li><b>Output:</b> Scalar loss value</li>
     * </ul>
     * 
     * @param input input tensor
     * @param target target tensor
     * @param reduction reduction method: "mean", "sum", or "none"
     * @return computed MSE loss
     */
    public static GradTensor mseLoss(GradTensor input, GradTensor target, String reduction) {
        return tech.kayys.tafkir.nn.loss.MSELoss.builder()
            .reduction(reduction)
            .build()
            .forward(input, target);
    }
    
    /**
     * Computes MSE loss with mean reduction.
     * 
     * @param input input tensor
     * @param target target tensor
     * @return computed MSE loss
     */
    public static GradTensor mseLoss(GradTensor input, GradTensor target) {
        return mseLoss(input, target, "mean");
    }
    
    /**
     * Computes L1 Loss (Mean Absolute Error).
     * 
     * @param input input tensor
     * @param target target tensor
     * @param reduction reduction method: "mean", "sum", or "none"
     * @return computed L1 loss
     */
    public static GradTensor l1Loss(GradTensor input, GradTensor target, String reduction) {
        return tech.kayys.tafkir.nn.loss.L1Loss.builder()
            .reduction(reduction)
            .build()
            .forward(input, target);
    }
    
    /**
     * Computes L1 loss with mean reduction.
     * 
     * @param input input tensor
     * @param target target tensor
     * @return computed L1 loss
     */
    public static GradTensor l1Loss(GradTensor input, GradTensor target) {
        return l1Loss(input, target, "mean");
    }
    
    /**
     * Applies 2D convolution operation.
     * 
     * <h3>Shape</h3>
     * <ul>
     *   <li><b>Input:</b> {@code [batchSize, inChannels, height, width]}</li>
     *   <li><b>Weight:</b> {@code [outChannels, inChannels/groups, kernelHeight, kernelWidth]}</li>
     *   <li><b>Bias:</b> {@code [outChannels]} (optional)</li>
     *   <li><b>Output:</b> {@code [batchSize, outChannels, outHeight, outWidth]}</li>
     * </ul>
     * 
     * @param input input tensor
     * @param weight convolution kernels
     * @param bias bias vector (optional)
     * @param stride stride for convolution (default: 1)
     * @param padding padding to add (default: 0)
     * @param dilation dilation factor (default: 1)
     * @param groups number of grouped connections (default: 1)
     * @return convolved tensor
     */
    public static GradTensor conv2d(GradTensor input, GradTensor weight, GradTensor bias,
                                    int stride, int padding, int dilation, int groups) {
        return tech.kayys.tafkir.nn.cnn.Conv2d.functionalConv2d(
            input, weight, bias, stride, padding, dilation, groups);
    }
    
    /**
     * Applies 2D convolution with default parameters.
     * 
     * @param input input tensor
     * @param weight convolution kernels
     * @param bias bias vector (optional)
     * @return convolved tensor
     */
    public static GradTensor conv2d(GradTensor input, GradTensor weight, GradTensor bias) {
        return conv2d(input, weight, bias, 1, 0, 1, 1);
    }
    
    /**
     * Applies max pooling over a 2D spatial input.
     * 
     * @param input input tensor of shape {@code [batchSize, channels, height, width]}
     * @param kernelSize size of the pooling window
     * @param stride stride of the pooling operation (default: kernelSize)
     * @param padding padding to add (default: 0)
     * @param dilation dilation factor (default: 1)
     * @return pooled tensor
     */
    public static GradTensor maxPool2d(GradTensor input, int kernelSize, int stride, 
                                       int padding, int dilation) {
        return tech.kayys.tafkir.nn.cnn.MaxPool2d.functionalMaxPool2d(
            input, kernelSize, stride, padding, dilation);
    }
    
    /**
     * Applies max pooling with default stride.
     * 
     * @param input input tensor
     * @param kernelSize size of the pooling window
     * @return pooled tensor
     */
    public static GradTensor maxPool2d(GradTensor input, int kernelSize) {
        return maxPool2d(input, kernelSize, kernelSize, 0, 1);
    }
    
    /**
     * Applies average pooling over a 2D spatial input.
     * 
     * @param input input tensor of shape {@code [batchSize, channels, height, width]}
     * @param kernelSize size of the pooling window
     * @param stride stride of the pooling operation (default: kernelSize)
     * @param padding padding to add (default: 0)
     * @param countIncludePad whether to include padding in averaging
     * @return pooled tensor
     */
    public static GradTensor avgPool2d(GradTensor input, int kernelSize, int stride,
                                       int padding, boolean countIncludePad) {
        return tech.kayys.tafkir.nn.cnn.AvgPool2d.functionalAvgPool2d(
            input, kernelSize, stride, padding, countIncludePad);
    }
    
    /**
     * Applies average pooling with default stride.
     * 
     * @param input input tensor
     * @param kernelSize size of the pooling window
     * @return pooled tensor
     */
    public static GradTensor avgPool2d(GradTensor input, int kernelSize) {
        return avgPool2d(input, kernelSize, kernelSize, 0, true);
    }
    
    /**
     * Flattens a tensor from startDim to endDim.
     * 
     * @param input input tensor
     * @param startDim first dimension to flatten (inclusive)
     * @param endDim last dimension to flatten (inclusive)
     * @return flattened tensor
     */
    public static GradTensor flatten(GradTensor input, int startDim, int endDim) {
        return input.flatten(startDim, endDim);
    }
    
    /**
     * Flattens all dimensions of a tensor.
     * 
     * @param input input tensor
     * @return flattened 1D tensor
     */
    public static GradTensor flatten(GradTensor input) {
        return input.flatten(0, input.shape().length - 1);
    }
    
    /**
     * Reshapes a tensor to the specified shape.
     * 
     * @param input input tensor
     * @param shape target shape
     * @return reshaped tensor
     */
    public static GradTensor reshape(GradTensor input, long... shape) {
        return input.reshape(shape);
    }
    
    /**
     * Transposes two dimensions of a tensor.
     * 
     * @param input input tensor
     * @param dim0 first dimension to swap
     * @param dim1 second dimension to swap
     * @return transposed tensor
     */
    public static GradTensor transpose(GradTensor input, int dim0, int dim1) {
        return input.transpose(dim0, dim1);
    }
    
    /**
     * Permutes dimensions of a tensor.
     * 
     * @param input input tensor
     * @param dims new order of dimensions
     * @return permuted tensor
     */
    public static GradTensor permute(GradTensor input, int... dims) {
        return input.permute(dims);
    }
    
    /**
     * Concatenates tensors along a dimension.
     * 
     * @param tensors array of tensors to concatenate
     * @param dim dimension to concatenate along
     * @return concatenated tensor
     */
    public static GradTensor cat(GradTensor[] tensors, int dim) {
        return tensors[0].cat(tensors, dim);
    }
    
    /**
     * Stacks tensors along a new dimension.
     * 
     * @param tensors array of tensors to stack
     * @param dim dimension to insert for stacking
     * @return stacked tensor
     */
    public static GradTensor stack(GradTensor[] tensors, int dim) {
        return tensors[0].stack(tensors, dim);
    }
    
    /**
     * Splits a tensor into chunks along a dimension.
     * 
     * @param input input tensor
     * @param splitSize size of each chunk (last chunk may be smaller)
     * @param dim dimension to split along
     * @return array of split tensors
     */
    public static GradTensor[] split(GradTensor input, int splitSize, int dim) {
        return input.split(splitSize, dim);
    }
    
    /**
     * Selects elements from input based on index tensor along a dimension.
     * 
     * @param input input tensor
     * @param index index tensor
     * @param dim dimension to select from
     * @return selected tensor
     */
    public static GradTensor indexSelect(GradTensor input, GradTensor index, int dim) {
        return input.indexSelect(index, dim);
    }
    
    /**
     * Embeds categorical indices using an embedding matrix.
     * 
     * <h3>Shape</h3>
     * <ul>
     *   <li><b>Input:</b> {@code [...]} with integer indices in [0, numEmbeddings-1]</li>
     *   <li><b>Weight:</b> {@code [numEmbeddings, embeddingDim]}</li>
     *   <li><b>Output:</b> {@code [..., embeddingDim]}</li>
     * </ul>
     * 
     * @param input indices tensor
     * @param weight embedding matrix
     * @return embedded tensor
     */
    public static GradTensor embedding(GradTensor input, GradTensor weight) {
        return tech.kayys.tafkir.nn.layer.Embedding.functionalEmbedding(input, weight);
    }
    
    /**
     * Adds positional encodings to input embeddings using rotary embeddings.
     * 
     * @param input input tensor with embeddings
     * @param cosPre cosine values precomputed for positions
     * @param sinPre sine values precomputed for positions
     * @return tensor with rotary positional encodings applied
     */
    public static GradTensor rotaryEmbed(GradTensor input, GradTensor cosPre, GradTensor sinPre) {
        return tech.kayys.tafkir.nn.layer.RotaryEmbedding.functionalRotaryEmbed(input, cosPre, sinPre);
    }
    
    /**
     * Computes scaled dot-product attention.
     * <p>
     * {@code Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V}
     * 
     * <h3>Shape</h3>
     * <ul>
     *   <li><b>Query:</b> {@code [batchSize, seqLenQ, numHeads, headDim]}</li>
     *   <li><b>Key:</b> {@code [batchSize, seqLenK, numHeads, headDim]}</li>
     *   <li><b>Value:</b> {@code [batchSize, seqLenV, numHeads, headDim]}</li>
     *   <li><b>Mask:</b> {@code [seqLenQ, seqLenK]} or broadcastable (optional)</li>
     *   <li><b>Output:</b> {@code [batchSize, seqLenQ, numHeads, headDim]}</li>
     * </ul>
     * 
     * @param query query tensor
     * @param key key tensor
     * @param value value tensor
     * @param mask attention mask (optional, can be null)
     * @param dropoutP dropout probability (default: 0.0)
     * @param isCausal whether to use causal masking (default: false)
     * @param scale scaling factor (if null, uses 1/sqrt(headDim))
     * @return attention output
     */
    public static GradTensor scaledDotProductAttention(
            GradTensor query, GradTensor key, GradTensor value,
            GradTensor mask, double dropoutP, boolean isCausal, Double scale) {
        return MultiHeadAttention.functionalAttention(
            query, key, value, mask, dropoutP, isCausal, scale);
    }
    
    /**
     * Computes scaled dot-product attention with default parameters.
     * 
     * @param query query tensor
     * @param key key tensor
     * @param value value tensor
     * @param mask attention mask (optional)
     * @param isCausal whether to use causal masking
     * @return attention output
     */
    public static GradTensor scaledDotProductAttention(
            GradTensor query, GradTensor key, GradTensor value,
            GradTensor mask, boolean isCausal) {
        return scaledDotProductAttention(query, key, value, mask, 0.0, isCausal, null);
    }
    
    /**
     * Computes scaled dot-product attention without mask.
     * 
     * @param query query tensor
     * @param key key tensor
     * @param value value tensor
     * @return attention output
     */
    public static GradTensor scaledDotProductAttention(
            GradTensor query, GradTensor key, GradTensor value) {
        return scaledDotProductAttention(query, key, value, null, false);
    }
}
