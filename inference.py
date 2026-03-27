from test import ContinuousBagOfWords, get_vocab_size
from pathlib import Path
from typing import Optional


def newest_file_with_ext(directory: str, ext: str) -> Optional[Path]:
    """
    Return the newest file in `directory` with extension `ext` (e.g. '.txt' or 'txt').
    Returns None if no matching files found.
    """
    p = Path(directory)
    if not ext.startswith('.'):
        ext = '.' + ext
    files = (f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ext.lower())
    try:
        return max(files, key=lambda f: f.stat().st_mtime)
    except ValueError:
        return None


def main():
    vsize = get_vocab_size()
    cbow = ContinuousBagOfWords(
        dictionary_size=vsize,
        hidden_size=vsize//20
    )
    
    path = "checkpoints"
    newest_model = newest_file_with_ext(path, ".wght")

    print("Loading", newest_model)

    if newest_model is None:
        raise RuntimeError("No models found in checkpoints!")

    cbow.load_weights(
        checkpoint_path=newest_model._str
    )

    # discard layers after first AveragingLinear

    # do inference


if __name__ == "__main__":
    main()