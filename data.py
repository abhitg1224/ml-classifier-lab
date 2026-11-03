"""Load and validate the Dry Bean data before splitting or evaluation."""

import csv
import hashlib
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split


SOURCE_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
FEATURES = (
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
)
LABELS = ("BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA")
TARGET = "Class"
COLUMNS = (*FEATURES, TARGET)
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_TEST_ROWS = 10_000


def validate_columns(columns) -> None:
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate column names are not allowed.")
    missing = set(COLUMNS) - set(columns)
    extra = set(columns) - set(COLUMNS)
    if missing or extra:
        raise ValueError(
            f"Incorrect columns. Missing: {sorted(missing)}; unexpected: {sorted(extra)}."
        )


def validate_data(frame: pd.DataFrame, *, max_rows: int | None = None) -> pd.DataFrame:
    validate_columns(frame.columns)
    if frame.empty:
        raise ValueError("At least one labeled observation is required.")
    if max_rows is not None and len(frame) > max_rows:
        raise ValueError(f"The CSV must contain at most {max_rows:,} rows.")
    if frame.isna().any().any():
        raise ValueError("Missing feature values or target labels are not allowed.")

    clean = frame.loc[:, list(COLUMNS)].copy()
    for feature in FEATURES:
        try:
            clean[feature] = pd.to_numeric(clean[feature], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{feature} must contain numeric values.") from error
    values = clean.loc[:, list(FEATURES)].to_numpy()
    if np.iscomplexobj(values):
        raise ValueError("Features must contain real numbers.")
    values = values.astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite; NaN and infinity are not allowed.")
    if (np.abs(values) > np.finfo(np.float32).max).any():
        raise ValueError("Feature values exceed the supported float32 range.")

    unknown = set(clean[TARGET]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown Class labels: {sorted(map(str, unknown))}.")
    return clean


def parse_csv(payload: bytes, *, max_rows: int | None = MAX_TEST_ROWS) -> pd.DataFrame:
    if len(payload) > MAX_CSV_BYTES:
        raise ValueError("The CSV exceeds the 10 MiB limit.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("The CSV must be UTF-8 encoded.") from error

    try:
        reader = csv.reader(StringIO(text), strict=True)
        header = next(reader, None)
        if not header:
            raise ValueError("The CSV needs a header and labeled observations.")
        validate_columns(header)
        rows = []
        for record_number, record in enumerate(reader, start=2):
            if not record:
                continue
            if len(record) != len(header):
                raise ValueError(f"CSV record {record_number} has an incorrect field count.")
            if max_rows is not None and len(rows) >= max_rows:
                raise ValueError(f"The CSV must contain at most {max_rows:,} rows.")
            rows.append(record)
    except csv.Error as error:
        raise ValueError(f"Malformed CSV: {error}") from error
    return validate_data(pd.DataFrame(rows, columns=header), max_rows=max_rows)


def download_dataset(destination: Path) -> pd.DataFrame:
    if destination.exists():
        raise FileExistsError(f"Source already exists: {destination}.")
    with urlopen(SOURCE_URL, timeout=90) as response:
        payload = response.read(MAX_CSV_BYTES + 1)
    if len(payload) > MAX_CSV_BYTES:
        raise ValueError("The UCI archive exceeds the 10 MiB limit.")
    with ZipFile(BytesIO(payload)) as archive:
        members = [
            item for item in archive.infolist()
            if item.filename.rsplit("/", 1)[-1] == "Dry_Bean_Dataset.arff"
        ]
        if len(members) != 1:
            raise ValueError("Expected exactly one Dry_Bean_Dataset.arff in the UCI archive.")
        if members[0].file_size > MAX_CSV_BYTES:
            raise ValueError("The dataset exceeds the 10 MiB limit.")
        text = archive.read(members[0]).decode("utf-8-sig")

    records, _ = arff.loadarff(StringIO(text))
    frame = pd.DataFrame(records)
    frame[TARGET] = frame[TARGET].str.decode("utf-8")
    clean = validate_data(frame)
    if len(clean) < 500 or set(clean[TARGET]) != set(LABELS):
        raise ValueError("The source must contain at least 500 rows and all seven classes.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(destination, index=False, lineterminator="\n", mode="x")
    return clean


def data_digest(frame: pd.DataFrame) -> str:
    csv_text = frame.loc[:, list(COLUMNS)].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def split_dataset(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    clean = validate_data(source)
    if len(clean) < 500:
        raise ValueError("The dataset must contain at least 500 observations.")
    if set(clean[TARGET]) != set(LABELS):
        raise ValueError("The dataset must include all seven classes.")

    conflicts = clean.groupby(list(FEATURES))[TARGET].nunique()
    if (conflicts > 1).any():
        raise ValueError("Identical feature vectors have conflicting labels.")
    unique = clean.drop_duplicates().reset_index(drop=True)
    if len(unique) < 500:
        raise ValueError("At least 500 unique observations are required after deduplication.")
    if unique[TARGET].value_counts().min() < 5:
        raise ValueError("Each class requires at least five distinct observations.")

    training, testing = train_test_split(
        unique, test_size=TEST_SIZE, stratify=unique[TARGET], random_state=RANDOM_STATE
    )
    training = training.reset_index(drop=True)
    testing = testing.reset_index(drop=True)
    overlap = training.merge(testing, on=list(FEATURES), how="inner")
    if not overlap.empty:
        raise ValueError("The training and test partitions share feature vectors.")
    report = {
        "source_rows": len(clean),
        "duplicates_removed": len(clean) - len(unique),
        "unique_rows": len(unique),
        "training_rows": len(training),
        "test_rows": len(testing),
        "features": list(FEATURES),
        "labels": list(LABELS),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "feature_overlap": len(overlap),
        "source_sha256": data_digest(clean),
        "training_sha256": data_digest(training),
        "test_sha256": data_digest(testing),
        "class_counts": {
            label: {
                "training": int((training[TARGET] == label).sum()),
                "test": int((testing[TARGET] == label).sum()),
            }
            for label in LABELS
        },
    }
    return training, testing, report
