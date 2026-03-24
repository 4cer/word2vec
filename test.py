import numpy as np
import csv, argparse, json
import types
from tqdm import tqdm


from com import model, layer, optimizer, loss


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
        train_path=r"dataset\processed\train.csv",
        test_path=r"dataset\processed\test.csv"
):
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
        vocab_path=r"dataset\processed\vocab.json",
) -> int:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab["vocab_size"]


def to_one_hot(indices: np.ndarray, vocab_size: int) -> np.ndarray:
    """Convert a flat array of word indices (w,) into one-hot rows (w, v, 1).

    Each integer index becomes a 1-hot vector of length vocab_size, then gets
    an extra size-1 axis so the shape matches what _forward() expects.
    """
    w = len(indices)
    one_hot = np.zeros((w, vocab_size, 1), dtype=np.float32)
    one_hot[np.arange(w), indices, 0] = 1.0
    return one_hot


def accuracy(cbow: ContinuousBagOfWords, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of samples where argmax(output) == true label."""
    correct = 0
    for indices, label in zip(X, y):
        x_hot = to_one_hot(indices, cbow.dictionary_size)
        pred = cbow(x_hot)
        if np.argmax(pred) == label:
            correct += 1
    return correct / len(y)


def train(
        cbow: ContinuousBagOfWords,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_verify: np.ndarray,
        y_verify: np.ndarray,
        batch_size: int = 50
) -> None:
    print("Starting training...")
    vocab_size = cbow.dictionary_size
    n_samples  = len(x_train)

    opt = optimizer.SGD(
        model=cbow,
        loss=loss.CategoricalCrossEntropy(),
        max_epochs=200,
        learning_rate=0.1,
    )
    
    opt.build_graph_once()

    epoch = 0
    best_val_accuracy = 0

    while opt.max_epochs < 0 or epoch < opt.max_epochs:
        epoch_loss = 0.0
 
        perm = np.random.permutation(n_samples)
        
        for batch_start in tqdm(range(0, n_samples, batch_size)):
            batch_indices = perm[batch_start : batch_start + batch_size]
 
            # 1. Prepare batch
            x_hot = np.stack([to_one_hot(x_train[i], vocab_size) for i in batch_indices])
            label = np.zeros((len(batch_indices), vocab_size), dtype=np.float32)
            label[np.arange(len(batch_indices)), y_train[batch_indices]] = 1.0
 
            # 2. SGD step for batch
            sample_loss = opt.propagate(x_hot, label)
            epoch_loss += sample_loss

        val_acc = accuracy(cbow, x_verify, y_verify)
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            checkpoint = f"./checkpoints/checkpoint_{epoch:06}_{val_acc:.2f}.wght"
            print("Saving checkpoint", checkpoint)
            opt.model.save_weights_fp32(f"./checkpoints/checkpoint_{epoch:06}_{val_acc:.2f}.wght")

        epoch += 1

        # Report every 10 epochs (and always on the first)
        if epoch == 1 or epoch % 10 == 0:
            train_acc = accuracy(cbow, x_train, y_train)
            avg_loss = epoch_loss / n_samples
            # val_acc   = accuracy(cbow, x_verify, y_verify)
            print(
                f"[epoch {epoch:>4}]  "
                f"loss: {avg_loss:.4f}  "
                f"train_acc: {train_acc:.3f}  "
                f"val_acc: {val_acc:.3f}"
            )


def inference_tests(
        cbow: ContinuousBagOfWords,
        x_verify: np.ndarray,
        y_verify: np.ndarray,
) -> None:
    acc = accuracy(cbow, x_verify, y_verify)
    print(f"\nFinal validation accuracy: {acc:.4f}")


def vector_size_test(vocab_size: int, cbow: ContinuousBagOfWords):
    """Throw an exception upon input shape mismatch.

    Args:
        vocab_size (int): Amount of words in vocabulary.
        cbow (ContinuousBagOfWords): Model reference.
    """
    cbow.print_layers()
    print("Shape testing...", end=" ")

    v1 = np.random.uniform(0.0, 1.0, (2, vocab_size, 1))
    y1: np.ndarray = cbow(v1)
    assert y1.shape == (vocab_size, 1)

    v2 = np.random.uniform(0.0, 1.0, (5, 2, vocab_size, 1))
    y2: np.ndarray = cbow(v2)
    assert y2.shape == (5, vocab_size, 1)

    print("[ OK ]")


def main():
    vsize = get_vocab_size()

    cbow = ContinuousBagOfWords(
        dictionary_size=vsize
    )

    vector_size_test(vsize, cbow)

    X_train, y_train, X_test, y_test = load_dataset()

    train(cbow, X_train, y_train, X_test, y_test)

    inference_tests(cbow, X_test, y_test)


if __name__ == "__main__":
    main()
