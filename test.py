from com import model, layer


class SkipGram(model.IModel):
    def __init__(self) -> None:
        self.linear1 = layer.Linear(3000, 300, False)
        self.linear2 = layer.Linear(300,3000, False)
        self.softmax = layer.SoftMax()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

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
