Modular Word2Vec
===

# About the project
Modular approach to defining, training and running inference, using PyTorch-like
declarative structure. Primary goal is extensibility for future layer types,
activation functions and loss functions.

# About the model
A Continuous Bag of Words (CBOW) model is implemented as an example. This uses
the autoencoder architecture. In training, the context of a word to be predicted
(n surrounding words) are fed in parallel through the first layer, which is
smaller than the size of the dictionary and averaged. The resulting vector is
then fed through to another layer the size of the original dictionary and then
a softmax activation layer.

After training is complete, the output layer and activation is discarded, and
the result of the first hidden layer is the embedding. While the loss function
adequately measures reconstruction quality, the quality of embeddings has to be
assessed separately.

## Derivation of backpropagation gradients for CBOW

[Derivation process](DERIVATION.md)

# Features
- Modularity, allowing further addition of layer types, schedulers, activation
  functions and loss functions.
- Pure numpy implementation.
- Declarative interface inspired by PyTorch.
- Stochastic Gradient Descent (SGD) optimizer with collapsed CCE+Softmax
  gradient and batch support.
- Schedulers
    - LinearScheduler: decrease LR over time as training progresses, up
      to selected epoch.
    - PlateauScheduler: decrease LR as loss plateaus, with configurable
      patience.
- Saving and loading weights to binary checkpoint files (`.wght` format).
- Automatic best-checkpoint saving during training (by validation accuracy).
- Backprop graph building with merging of common last activation/loss
  combinations.

# Requirements
- conda environment (`.conda/`).
- Jupyter environment.
- numpy.
- Git with LFS for submodule dataset repository **or alternatively**
    - The dataset can be sorced manually (linked in Used Sources)
    - Another compatible dataset with normalised case and no punctuation may be
        used.
  

# Running
## Environment setup
### 0. Installing LFS
```
git lfs install
```

### 1. Pulling repository contents into current directory
```
git clone https://github.com/4cer/word2vec.git .
```
If LFS is installed, the submodule dataset will be pulled automatically into
`./dataset/repo` as part of the cloning operation.

Otherwise it will be necessary to return to step 1, then here and run:
```
git submodule update dataset/repo
```

### 2a. Quick setup
```
conda env create -f environment.yml --prefix ./.conda
```
 
### 2b. Exact reproduction of a known-good environment
```
conda env create -f environment.lock.yml --prefix ./.conda
```

## 3. Data preparation
Run `data_prep.ipynb` to tokenise the raw corpus, build the vocabulary, and
write the windowed context pairs to `dataset/processed/train.csv`, `test.csv`,
and `vocab.json`. The CSV columns are `x` (space-separated context word
indices) and `y` (centre word index).

## 4. Training
Run `test.py` from the repository root:
```
python test.py
```
The script loads the processed dataset, instantiates a `ContinuousBagOfWords`
model, and trains it with SGD for the configured number of epochs, printing loss
and accuracy every 10 epochs. The best checkpoint by validation accuracy is
saved automatically to `./checkpoints/` after each epoch that improves on the
previous best. Training can be safely interrupted with `Ctrl+C` at any time;
the most recent best checkpoint will be preserved.

## 5. Inference
Load a saved checkpoint with `model.load_weights_fp32(path)`, then discard
`linear2` and `softmax` — the rows of `linear1.weights` are the trained word
embeddings, indexed by vocabulary ID. Pass any sequence of context word indices
through `linear1` to retrieve their embedding vectors.

# Project Structure
```
4cer-word2vec/
├── README.md               — This file
├── DERIVATION.md           — Step-by-step math for CBOW backpropagation gradients
├── data_prep.ipynb         — Jupyter notebook: tokenise corpus, build vocab, write CSV pairs
├── test.py                 — Main entry point: defines ContinuousBagOfWords, training loop,
│                             shape tests, and inference evaluation
├── environment.yml         — Minimal conda environment spec (Python 3.11, numpy, jupyter)
├── environment.lock.yml    — Fully pinned conda environment for exact reproduction
├── checkpoints/
│   └── structure.txt       — Binary format spec for .wght checkpoint files
└── com/                    — Core library package
    ├── __init__.py         — Re-exports all submodules (layer, loss, model, optimizer, scheduler)
    ├── layer.py            — ILayer ABC; Linear, AveragingLinear, ReLU, SoftMax, Sigmoid
    ├── loss.py             — ILossFunction ABC; CategoricalCrossEntropy, CrossEntropy stubs
    ├── model.py            — IModel ABC; checkpoint save/load (.wght binary format)
    ├── optimizer.py        — IOptimizer ABC; SGD with collapsed CCE+Softmax gradient
    └── scheduler.py        — IScheduler ABC; LinearScheduler, PlateauScheduler
```

### Key file descriptions
 
**`test.py`** — The main script and the only file you need to run for training. It
defines `ContinuousBagOfWords` (a concrete `IModel` subclass), wires together the
dataset loader, one-hot encoder, SGD optimizer, and plateau scheduler, and runs the
full train → checkpoint → evaluate loop. Run this from the repository root after
data preparation.
 
**`com/model.py`** — Foundation of the framework. `IModel` manages the ordered layer
registry, caching hooks, graph-tracing hooks, and checkpoint (de)serialisation in a
custom binary `.wght` format. All concrete models inherit from this.
 
**`com/layer.py`** — All layer types live here. `ILayer` defines the forward/caching/back
interface and the `LayerType` enum used for graph building and serialisation.
`AveragingLinear` is CBOW-specific: it averages context word vectors before
the matrix multiply, handling the CBOW pooling step implicitly.
 
**`com/optimizer.py`** — `SGD` builds a reversed layer graph once, then for each batch
runs forward, computes the collapsed CCE+Softmax gradient (skipping the numerically
expensive Jacobian), and updates weights via `einsum`-based batch averaging.
`COLLAPSE_TABLE` maps `(loss_type, last_layer_type)` pairs to their fused gradient
functions for extensibility.
 
**`com/scheduler.py`** — `PlateauScheduler` watches a chosen metric (accuracy or loss)
and multiplies the learning rate by `factor` after `patience` epochs without
improvement. `LinearScheduler` anneals the rate linearly to a target over a fixed
number of epochs.
 
**`data_prep.ipynb`** — Run once before training. Reads the raw text8 corpus, applies
frequency filtering (`MIN_FREQ = 5`) and Word2Vec subsampling, builds a JSON vocab
map, and writes windowed `(X, y)` pairs to `train.csv` and `test.csv`.
 
**`checkpoints/structure.txt`** — Documents the `.wght` binary format: magic bytes
`WGHT`, version, layer count, then per-layer records with type, shape, and raw
float32 weights. Activation layers write a zero-length record to keep indices aligned.

# Results
## PCA of embeddings
*(to be added)*

## Next steps
Possible performance improvements:
- Implement GPU training and inference via CUDA, Vulkan, ROCm, or OpenCL.
- Implement embedding quality evaluation (analogy tasks, nearest-neighbour
  cosine similarity) in a Jupyter notebook.
- Extend `COLLAPSE_TABLE` with additional fused gradient pairs.
Misc:
- Add biases to the `Linear` layer.

## Known Issues and Limitations
- Biases are not implemented in the `Linear` layer.
- `CrossEntropy` (binary) is a stub; only `CategoricalCrossEntropy` is
  implemented.

# TODO
- [x] Model weight init
    - [x] Random weights
    - [x] Weights from file
- [x] Save weights to file
- [x] Implement loss functions
- [x] Implement SGD Optimizer:
    - [x] Optimizer: Make graph on first model pass
        - [x] Model and Layer: register in graph
    - [x] Add loss and disable graph making
    - [x] Solve graph backwards, merging known combinations
    - [x] Save the graph for backprop
- [x] Implement Linear Scheduler
- [x] Implement Plateau Scheduler
- [x] Implement data preparation via Jupyter notebook
- [x] Implement main script flow
- [x] Implement batching
    - [x] Load dataset into numpy arrays
    - [x] Train using SGD
    - [ ] Test
- [ ] Unless static graph enabled, rebuild at the start of each
      optimizer.propagate
- [ ] Central script to dispatch to tests, training, inference or combination
      using argparse.
- [ ] Implement embedding quality check in Jupyter
- [ ] Implement biases in Linear layer

# Used Sources
- text8 dataset — [HuggingFace mirror](https://huggingface.co/roshbeed/text8-dataset)
- [micrograd by Andrej Karpathy](https://github.com/karpathy/micrograd)
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
