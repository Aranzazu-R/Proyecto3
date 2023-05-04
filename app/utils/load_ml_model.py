import pickle
from typing import Any


def load_ml_model(path: str) -> Any:

    with open(path, "rb") as file:
        deserialized_model = pickle.load(file, fix_imports=True)

    return deserialized_model
