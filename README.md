Modular Word2Vec
===

# About the project
Modular approach to defining, training and running inference, using PyTorch-like
declarative structure. Primary goal is extensibility for future layer types,
activation functions and loss functions.

# About the model
A Continuous Bag of Words (CBOW) model is implemented as an example. This uses
the autoencoder architecture. In training, the context of a word to be predicted
(n surrounding words) are fed fed in parallel, through the first layer, which is
smaller than the size of the dictionary and averaged. The resulting vector is
then fed through to another layer the size of the original dictionary and then
softmax activation layer.

After training is complete. the output layer and activation is discarded, and
the result of the first hidden layer is the embedding. While the loss function
adequately measures reconstruction quality, the quality of embeddings has to be
assessed separately.

## Derivation of backpropagation gradients for CBOW

[Derivation process](DERIVATION.md)

# Features
- Modularity, allowing for adding further activation layer types and activation
functions later.
- Pure numpy implementation.
- Declarative interface inspired by PyTorch.
- Stochastic Gradient Descent (SGD) optimizer with collapsed CCE+Softmax
gradient.
- Schedulers
    - LinearScheduler: decrease LR over time as training progresses, up
    to selected epoch.
    - (TODO) PlateuScheduler: decrease LR as loss plateaus, with configurable
    patience.
- Saving and loading weights to binary checkpoint files (`.wght` format).
- Backprop graph building with merging of common last activation/loss
combinations.

# Requirements
- .conda environment.
- Jupyter environment.
- numpy.
- Git with LFS for submodule repository of dataset
    - Or sourcing a similar dataset with normalized case and no punctuation.

# Running
## Environment setup
### Quick setup
`conda env create -f environment.yml`

### Exact reproduction of a known-good environment
`conda env create -f environment.lock.yml`

## Data preparation
Run the data preparation notebook to tokenize the raw corpus, build the
vocabulary, and write the windowed context pairs to
`dataset/processed/train.csv`, `test.csv`, and `vocab.json`. The CSV columns
are `x` (space-separated context word indices) and `y` (centre word index).

## Training
Run `test.py` from the repository root. The script loads the processed dataset,
instantiates a `ContinuousBagOfWords` model, and trains it with SGD for the
configured number of iterations, printing loss and accuracy every 10 epochs.
Weights are not saved automatically; call `model.save_weights_fp32(path)` after
training to write a checkpoint.

## Inference
Load a saved checkpoint with `model.load_weights_fp32(path)`, then discard
`linear2` and `softmax` — the rows of `linear1.weights` are the trained word
embeddings, indexed by vocabulary ID. Pass any sequence of context word indices
through `linear1` to retrieve their embedding vectors.

# Results
## PCA of embeddings
Walla

# Known Issues and Limitations
- `Linear.forward` writes the matmul result back into the input buffer
(`out=input`), so `layer.cache` captures the layer's *output*, not its *input*.
The weight-gradient outer product in `SGD.propagate` therefore uses the output
as a stand-in for the input. This is consistent internally but will diverge from
the mathematically correct gradient once non-linear activations precede a Linear
layer. Fix: cache the pre-matmul input separately.
- `SoftMax.forward_caching` currently computes a sigmoid instead of a softmax
(copy-paste error from `Sigmoid`). Since the CCE+Softmax gradient is collapsed
and `SoftMax.back` is never called in normal operation, this does not affect
training, but it would produce wrong outputs if the cache were inspected
directly.
- `PlateauScheduler` is not yet implemented.
- Biases are not implemented in the `Linear` layer.

# TODO
- <input type="checkbox" disabled checked> Model weight init
    - <input type="checkbox" disabled checked> Random weights
    - <input type="checkbox" disabled checked> Weights from file
- <input type="checkbox" disabled checked> Save weights to file
- <input type="checkbox" disabled checked> Implement loss functions
- <input type="checkbox" disabled checked> Implement SGD Optimizer:
    - <input type="checkbox" disabled checked> Optimizer: Make graph on first
    model pass
        - <input type="checkbox" disabled checked> Model and Layer: register in
        graph
    - <input type="checkbox" disabled checked> Add loss and disable graph making
    - <input type="checkbox" disabled checked> Solve graph backwards, merging
    known combinations
    - <input type="checkbox" disabled checked> Save the graph for backprop
- <input type="checkbox" disabled checked> Implement Linear Scheduler
- <input type="checkbox" disabled> Implement Plateau Scheduler
- <input type="checkbox" disabled checked> Implement data preparation via
Jupyter notebook
- <input type="checkbox" disabled checked> Implement main script flow
    - <input type="checkbox" disabled checked> Load dataset into numpy arrays
    - <input type="checkbox" disabled checked> Train using SGD
    - <input type="checkbox" disabled> Test
- <input type="checkbox" disabled> Implement embedding quality check in Jupyter
- <input type="checkbox" disabled> Implement biases in Linear layer

# Used Sources
- text8 dataset
- [micrograd by Andrej Carpathy](https://github.com/karpathy/micrograd)
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
