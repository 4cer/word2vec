import numpy as np
import csv, json
from tqdm import tqdm


from com import model, layer, optimizer, loss, scheduler


class ContinuousBagOfWords(model.IModel):
    def __init__(
            self,
            dictionary_size: int,
            hidden_size: int = 512
    ) -> None:
        super().__init__()
        self.dictionary_size = dictionary_size
        self.linear1 = layer.AveragingLinear(self, dictionary_size, hidden_size)
        self.linear2 = layer.Linear(self, hidden_size, dictionary_size)
        self.softmax = layer.SoftMax(self)

        self.linear1.init_random(-0.5 / dictionary_size, 0.5 / dictionary_size)
        self.linear2.init_zeros()
        
    def _forward(self, x: np.ndarray):
        """Feed forward operation.

        Feeds a set of 1-hot vectors, or a batch of sets of 1-hot vectors 
        forward through the model. Importatly, window size is understood as
        number of surrounding words on EACH SIDE; x shape must be (b, w, v, 1)
        if batched or (w, v, 1) otherwise, where:

        - b: batch size
        - w: surrounding word cound (2x window size)
        - v: vocabulary size.

        Args:
            x (np.ndarray): Window-sized set of 1-hot vectors or batch thereof.

        Returns:
            np.ndarray: Pobabilities for each word in vocab to be the middle
                word.
        """
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    
    def print_layers(self):
        print(self.layers)


def load_dataset(
        train_path="dataset/processed/train.csv",
        test_path="dataset/processed/test.csv"
):
    """Loads processed dataset

    Retrieves both the training and test subsets of the dataset, packages them
    into numpy arrays and returns in predictable order.

    Expects a header and for each datapoint to be formatted as such:
    ```
        {x1 x2 x3 ... x_2w-2 x_2w-1 x_2w},{x_n}
    ```

    Args:
        train_path (regexp, optional): Relative or absolute path to the training
            fraction of data in csv format.
            Defaults to r"dataset/processed/train.csv".
        test_path (regexp, optional): Relative or absolute path to the test
            fraction of data in csv format.
            Defaults to r"dataset/processed/test.csv".

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Four numpy arrays:
        - first pair: X_train, Y_train
        - second pair: X_test, Y_test
    """
    print("Loading dataset...", end=" ")
    def read_csv(path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            X, y = [], []
            for row in reader:
                X.append(list(map(int, row["x"].split())))
                y.append(int(row["y"]))
        return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)

    X_train, y_train = read_csv(train_path)
    X_test,  y_test  = read_csv(test_path)

    print("[ OK ]")
    print(f"X_train {X_train.shape}  y_train {y_train.shape}")
    print(f"X_test  {X_test.shape}   y_test  {y_test.shape}")
    return X_train, y_train, X_test, y_test


def get_vocab_size(
        vocab_path="dataset/processed/vocab.json",
) -> int:
    """Read the vocab file to retrieve the vocab size."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab["vocab_size"]


def to_one_hot(indices: np.ndarray, vocab_size: int) -> np.ndarray:
    """Convert word indices to one-hot vectors.
     
    Supports single samples of size (w,) or batches of size (b, w), turning them
    into one-hot rows of shape (w, v, 1) and (b, w, v, 1) respectively, where:
    - b: batch size
    - w: window size * 2
    - v: vocabulary size
    
    Each integer index becomes a 1-hot vector of length vocab_size, then gets
    an extra size-1 axis so the shape matches what _forward() expects.

    Args:
        indices (np.ndarray): Either shape (w,) for a single sample or (b, w)
            for a batch.
        vocab_size (int): Length of each one-hot vector.

    Returns:
        np.ndarray: Shape (w, v, 1) for a single sample, or (b, w, v, 1) for
            a batch. dtype float32.
    """
    batched = indices.ndim == 2
    idx = indices if batched else indices[np.newaxis, :]   # (b, w)
    b, w = idx.shape
    one_hot = np.zeros((b, w, vocab_size, 1), dtype=np.float32)
    one_hot[np.arange(b)[:, None], np.arange(w)[None, :], idx, 0] = 1.0
    return one_hot if batched else one_hot[0]


def accuracy(
        cbow: ContinuousBagOfWords,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 512
) -> float:
    """Fraction of samples where argmax(output) == true label."""
    correct = 0

    for batch_start in range(0, len(X), batch_size):
        x_hot = to_one_hot(
            indices=X[batch_start:batch_start + batch_size],
            vocab_size=cbow.dictionary_size
        )
        pred_batch = cbow(x_hot) # (b, v, 1)
        pred_batch = np.argmax(pred_batch, axis=-2) # (b,1)
        label_batch = y[batch_start:batch_start + batch_size] # (b,)
        
        correct += np.sum(pred_batch == label_batch)

    return correct / len(y)


def train(
        cbow: ContinuousBagOfWords,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_verify: np.ndarray,
        y_verify: np.ndarray,
        batch_size: int = 50,
        max_epochs: int = 500
) -> None:
    """Execute training flow for the input model.

    In general, this function is designed to be safe to stop prematurely, after
    achieving the desired level of accuracy. Simply use Ctrl+C to stop, after
    a checkpoint of sufficient quality is saved.

    Args:
        cbow (ContinuousBagOfWords): _description_
        x_train (np.ndarray): Sets of indices of words in the vocabulary in
            window-sized neighborhood of y label. Training fraction.

        y_train (np.ndarray): Indices of words in the vocabulary serving as
            labels for the neighborhood sets in x. Training fraction.

        x_verify (np.ndarray): Sets of indices of words in the vocabulary in
            window-sized neighborhood of y label. Verification fraction.

        y_verify (np.ndarray): Indices of words in the vocabulary serving as
            labels for the neighborhood sets in x. Verification fraction.

        batch_size (int, optional): Amount of x,y pairs to average gradient
            changes across. Defaults to 50.

        max_epochs (int, optional): Amount of total runs through the training
            dataset to cut off at. Defaults to 500.
    """
    print("Starting training...")
    vocab_size = cbow.dictionary_size
    n_samples  = len(x_train)

    opt = optimizer.SGD(
        model=cbow,
        loss=loss.CategoricalCrossEntropy(),
        max_epochs=max_epochs,
        learning_rate=0.1,
    )

    sched = scheduler.PlateauScheduler(
        optimizer=opt,
        factor=0.1,
        threshold=1e-4,
        min_lr=1e-10,
        patience=10,
        verbosity=2,
        metric=scheduler.PlateauScheduler.PerformanceMetric.ACCURACY
    )
    
    opt.build_graph_once()

    epoch = 0
    best_val_accuracy = 0

    try:
        while opt.max_epochs < 0 or epoch < opt.max_epochs:
            epoch_loss = 0.0
    
            perm = np.random.permutation(n_samples)
            
            for batch_start in tqdm(range(0, n_samples, batch_size)):
                batch_indices = perm[batch_start : batch_start + batch_size]
    
                # 1. Prepare batch
                x_hot = to_one_hot(x_train[batch_indices], vocab_size)
                label = np.zeros((len(batch_indices), vocab_size), dtype=np.float32)
                label[np.arange(len(batch_indices)), y_train[batch_indices]] = 1.0
    
                # 2. SGD step for batch
                sample_loss = opt.propagate(x_hot, label)
                epoch_loss += sample_loss

            epoch += 1
            val_acc = accuracy(cbow, x_verify, y_verify)
            sched.step(accuracy=val_acc)
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                checkpoint = f"./checkpoints/checkpoint_{epoch:06}_{val_acc:.8f}.wght"
                print("Saving checkpoint", checkpoint)
                opt.model.save_weights_fp32(checkpoint)

            # Report every 10 epochs (and always on the first)
            if epoch == 1 or epoch % 10 == 0:
                train_acc = accuracy(cbow, x_train, y_train)
                avg_loss = epoch_loss / n_samples
                print(
                    f"[epoch {epoch:>4}]  "
                    f"loss: {avg_loss:.4f}  "
                    f"train_acc: {train_acc:.3f}  "
                    f"val_acc: {val_acc:.3f}"
                )
    except KeyboardInterrupt:
        print("Exitting on keyboard interrupt.")
        exit()


def vector_size_test(vocab_size: int, cbow: ContinuousBagOfWords):
    """Simple smoke test for in/out shapes of processed tensors.

    Tests only forward propagation. Fails fast upon mismatches.

    Args:
        vocab_size (int): Amount of words in vocabulary.
        cbow (ContinuousBagOfWords): Reference to model object.
    """
    cbow.print_layers()
    print("Shape testing...", end=" ", flush=True)

    v1 = np.random.uniform(0.0, 1.0, (2, vocab_size, 1))
    y1: np.ndarray = cbow(v1)
    assert y1.shape == (vocab_size, 1)

    v2 = np.random.uniform(0.0, 1.0, (5, 2, vocab_size, 1))
    y2: np.ndarray = cbow(v2)
    assert y2.shape == (5, vocab_size, 1)

    print("[ OK ]")

def accuracy_smoke_test(
        cbow: ContinuousBagOfWords
):
    print("Smoke testing batched accuracy...", end=" ", flush=True)
    shape = (1024,1)
    X = np.random.randint(1, cbow.dictionary_size, size=shape, dtype=np.int32)
    Y = np.zeros(shape, dtype=np.int32)
    Y[0:512] = X[0:512]
    acc = accuracy(
        cbow=cbow,
        X=X,
        y=Y,
        batch_size=64
    )

    assert acc == 0.5

    print("[ OK ]")


def inference_tests(
        cbow: ContinuousBagOfWords,
        x_verify: np.ndarray,
        y_verify: np.ndarray,
) -> None:
    """Test post-training accuracy.

    Used for the assessment of the final state of the model.

    Args:
        cbow (ContinuousBagOfWords): Reference to model object.

        x_verify (np.ndarray): Sets of indices of words in the vocabulary in
            window-sized neighborhood of y label. Verification fraction.

        y_verify (np.ndarray): Indices of words in the vocabulary serving as
            labels for the neighborhood sets in x. Verification fraction.
    """
    acc = accuracy(cbow, x_verify, y_verify)
    print(f"\nFinal validation accuracy: {acc:.4f}")


def main():
    vsize = get_vocab_size()

    cbow = ContinuousBagOfWords(
        dictionary_size=vsize,
        hidden_size=vsize//20
    )

    vector_size_test(vsize, cbow)

    accuracy_smoke_test(cbow=cbow)

    X_train, y_train, X_test, y_test = load_dataset()

    train(cbow, X_train, y_train, X_test, y_test)

    inference_tests(cbow, X_test, y_test)


if __name__ == "__main__":
    main()
