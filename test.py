import numpy as np
import csv, argparse, json
import types


from com import model, layer


class ContinuousBagOfWords(model.IModel):
    def __init__(
            self,
            dictionary_size: int,
            hidden_size: int = 512
    ) -> None:
        super().__init__()
        self.linear1 = layer.Linear(self, dictionary_size, hidden_size)
        self.linear2 = layer.Linear(self, hidden_size, dictionary_size)
        self.softmax = layer.SoftMax(self)
        
        self.linear1.init_random(-0.5 / dictionary_size, 0.5 / dictionary_size)
        self.linear2.init_zeros()
        
    def forward(self, x: np.ndarray):
        """Feed forward operation.

        Feeds a set of 1-hot vectors, or a batch of sets of 1-hot vectors 
        forward through the model. Importatly, window size is understood as
        number of surrounding words on EACH SIDE; x shape must be (b, w, 1, v)
        if batched or (w, 1, v) otherwise, where:

                - b - batch size
                - w - surrounding word cound (2x window size)
                - v - vocabulary size.

        Args:
            x (np.ndarray): Window-sized set of 1-hot vectors or batch thereof.

        Returns:
            np.ndarray: Pobabilities for each word in vocab to be the middle
            word.
        """
        x = self.linear1(x)
        x = x.mean(axis=-3)

        x = self.linear2(x)
        x = self.softmax(x)
        return x
    
    def print_layers(self):
        print(self.layers)


def load_dataset(
        train_path=r"dataset\processed\train.csv",
        test_path=r"dataset\processed\test.csv"
):
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

    print(f"X_train {X_train.shape}  y_train {y_train.shape}")
    print(f"X_test  {X_test.shape}   y_test  {y_test.shape}")
    return X_train, y_train, X_test, y_test


def get_vocab_size(
        vocab_path=r"dataset\processed\vocab.json",
) -> int:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab["vocab_size"]


def train(
        x_train,
        y_train,
        x_verify,
        y_verify
):
    pass


def inference_tests(
        x_verify,
        y_verify
):
    pass


def main():
    model = ContinuousBagOfWords(
        dictionary_size=get_vocab_size()
    )
    
    model.print_layers()
    exit()

    X_train, y_train, X_test, y_test = load_dataset()

    train(X_train, y_train, X_test, y_test)

    inference_tests(X_test, y_test)
    pass


if __name__ == "__main__":
    main()
