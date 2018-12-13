import pickle
import gzip
from pathlib import Path
import requests

def load_mnist():
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        print('Downloading MNIST data...')
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    print('Unziping data...')
    with gzip.open(PATH / FILENAME, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    return x_train, y_train, x_valid, y_valid
