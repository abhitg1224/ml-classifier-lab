"""Load only repository-controlled fitted pipelines with matching dependencies."""

import hashlib
from io import BytesIO
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.pipeline import Pipeline

from data import FEATURES, LABELS
from models import MODEL_FILES


def runtime_versions() -> dict:
    return {
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models(directory: Path) -> dict[str, Pipeline]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported saved-model manifest format.")
    if manifest["versions"] != runtime_versions():
        raise ValueError("Saved-model dependency versions differ. Install requirements.txt exactly.")
    if (
        manifest["features"] != list(FEATURES)
        or manifest["labels"] != list(LABELS)
        or set(manifest["models"]) != set(MODEL_FILES)
    ):
        raise ValueError("Saved-model metadata does not match the dataset schema.")

    verified = {}
    for name, filename in MODEL_FILES.items():
        entry = manifest["models"][name]
        if entry["filename"] != filename:
            raise ValueError(f"Unexpected artifact filename for {name}.")
        payload = (directory / filename).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"Artifact checksum mismatch for {name}. Restore the trusted file.")
        verified[name] = payload

    loaded = {}
    for name, payload in verified.items():
        model = joblib.load(BytesIO(payload))
        if (
            not isinstance(model, Pipeline)
            or list(model.feature_names_in_) != list(FEATURES)
            or len(model.classes_) != len(LABELS)
            or set(model.classes_) != set(LABELS)
        ):
            raise ValueError(f"Saved pipeline schema mismatch for {name}.")
        loaded[name] = model
    return loaded
