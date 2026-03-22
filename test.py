from com import Model, Layer, Linear, SoftMax


class SkipGram(Model):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        pass

def prepare_vocabulary() -> tuple[list[float], list[str]]:
    return (
        [],
        []
    )

def build_shuffled_sets() -> tuple[list[float], list[float], list[float], list[float]]:
    return (
        [],
        [],
        [],
        []
    )

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


if __name__ == "__main__":
    vocab_vecs, vocab_words = prepare_vocabulary()

    x_train, y_train, x_verify, y_verify = build_shuffled_sets()

    train(x_train, y_train, x_verify, y_verify)

    inference_tests(x_verify, y_verify)
