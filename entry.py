import argparse, pathlib


def main():
    parser = argparse.ArgumentParser(
        prog='Modular ML engine CBOW example',
        description=("Facilitates the training and testing of a CBOW word2vec"
         "model, as well as inference using the latest or specified checkpoint."
        " The order of operations is always `tests > training > inference >"
        " quality verification`"),
        epilog="For more info consult README.md"
    )

    group_action = parser.add_argument_group("Action")
    group_action.add_argument("-s", "--skip-smoke-tests",
                              action="store_true",
                              help=("Do not run smoke tests before"
                              " training/inference."))
    group_action.add_argument("-u", "--skip-unit-tests",
                              action="store_true",
                              help=("Do not run unit tests before"
                              " training/inference."))

    group_action.add_argument("-t", "--training",
                              action="store_true",
                              help=("Train a model for the defined layout."
                              " Defaults to False."))
    group_action.add_argument("-a", "--test-quality",
                              action="store_true",
                              help=("Test vector quality of embeddings using a"
                              " suite of metrics. Defaults to the same as"
                              " --training."))
    group_action.add_argument("-i", "--run-inference",
                              action="store_true",
                              help=("Perform inference on the latest"
                              " checkpoint compatible with the defined model."
                              " Alternatively, a specific checkpoint may be"
                              " specified using -c {path}"))

    group_action.add_argument("-m", "--checkpoint-metadata", type=pathlib.Path,
                              help=("Read and print just the metadata of"
                              " checkpoint at specified path."))

    group_files = parser.add_argument_group("Source files")
    group_files.add_argument("-d", "--dataset", type=pathlib.Path)
    group_files.add_argument("-c", "--checkpoint", type=pathlib.Path)
    group_files.add_argument("-f", "--input-tensor", type=pathlib.Path)
    group_files.add_argument("-o", "--output-tensor", type=pathlib.Path)

    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()