"""Explore saved Dry Bean classifiers using labeled test observations."""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from artifacts import file_digest, load_models
from data import FEATURES, LABELS, MAX_CSV_BYTES, MAX_TEST_ROWS, TARGET, parse_csv
from metrics import METRIC_NAMES, comparison_table, evaluate
from models import MODEL_FILES


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("classifier_app")


@st.cache_resource(max_entries=2, show_spinner=False)
def trusted_models(directory: str, signature: tuple) -> dict:
    return load_models(Path(directory))


@st.cache_data(max_entries=3, show_spinner=False)
def analyze(payload: bytes, directory: str, signature: tuple) -> tuple:
    frame = parse_csv(payload)
    models = trusted_models(directory, signature)
    return frame, {name: evaluate(model, frame) for name, model in models.items()}


def main() -> None:
    st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon=":seedling:", layout="wide")
    st.title("Dry Bean Classifier Lab")
    st.caption("Five classifiers | Six evaluation metrics | One shared test dataset")
    st.write(
        "Compare seven bean varieties using 16 numerical shape measurements. "
        "Evaluation uses saved training pipelines; uploaded observations are never used for fitting."
    )

    with st.sidebar:
        st.header("Evaluation controls")
        selected = st.selectbox("Classifier", list(MODEL_FILES), key="classifier")
        bundled = st.checkbox("Use bundled held-out test data", value=False, key="bundled")
        st.caption(
            "Upload a labeled CSV or enable the bundled holdout. An uploaded file takes precedence."
        )
        st.caption("Precision, Recall and F1: macro average. AUC: macro one-vs-rest.")

    st.subheader("Test observations")
    st.write(
        f"CSV columns must match the sample: {len(FEATURES)} numerical features and "
        f"the target `{TARGET}`. Maximum {MAX_TEST_ROWS:,} rows and "
        f"{MAX_CSV_BYTES // (1024 * 1024)} MiB; UTF-8 encoding."
    )
    uploaded = st.file_uploader("Upload labeled test CSV", type=["csv"], key="test_csv")
    try:
        sample = (ROOT / "test_data.csv").read_bytes()
        st.download_button(
            "Download test CSV", sample, file_name="test_data.csv",
            mime="text/csv", key="sample_download",
        )
        if uploaded is not None:
            payload = uploaded.getvalue()
            source = f"Uploaded: {uploaded.name}"
        elif bundled:
            payload = sample
            source = "Bundled held-out test data"
        else:
            st.info("Upload a test CSV or enable the bundled held-out test data to begin.")
            st.stop()
        directory = ROOT / "model"
        signature = tuple(
            (filename, file_digest(directory / filename))
            for filename in ("manifest.json", *MODEL_FILES.values())
        )
        with st.spinner("Evaluating all five saved classifiers..."):
            frame, results = analyze(payload, str(directory), signature)
    except (OSError, ValueError, KeyError) as error:
        LOGGER.exception("Test-data evaluation failed")
        st.error(f"Cannot evaluate this dataset: {error}")
        st.stop()

    st.caption(f"{source} | {len(frame):,} observations | {frame[TARGET].nunique()} observed classes")
    st.subheader(selected)
    result = results[selected]
    for column, metric in zip(st.columns(len(METRIC_NAMES)), METRIC_NAMES):
        value = result["metrics"][metric]
        column.metric(metric, "N/A" if value is None else f"{value:.6f}")
    if result["auc_explanation"]:
        st.warning(result["auc_explanation"])

    st.subheader("All-model comparison")
    st.caption("Every row below uses the same current test observations, not the training data.")
    table = comparison_table(results)
    st.dataframe(table.style.format("{:.6f}", na_rep="N/A"), width="stretch")
    st.download_button(
        "Download comparison CSV", table.to_csv(lineterminator="\n"),
        file_name="comparison.csv", mime="text/csv",
    )

    matrix_tab, report_tab, data_tab = st.tabs(
        ["Confusion matrix", "Classification report", "Data preview"]
    )
    with matrix_tab:
        st.caption("Rows: actual class. Columns: predicted class.")
        matrix = pd.DataFrame(
            result["confusion_matrix"],
            index=pd.Index(LABELS, name="Actual"),
            columns=pd.Index(LABELS, name="Predicted"),
        )
        st.dataframe(matrix, width="stretch")
    with report_tab:
        rows = [*LABELS, "macro avg", "weighted avg"]
        report = pd.DataFrame.from_dict(
            {label: result["classification_report"][label] for label in rows}, orient="index"
        )
        st.dataframe(
            report.style.format(
                {"precision": "{:.6f}", "recall": "{:.6f}", "f1-score": "{:.6f}", "support": "{:.0f}"}
            ), width="stretch",
        )
    with data_tab:
        st.caption("First 100 validated observations. Evaluation includes every uploaded row.")
        st.dataframe(frame.head(100), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
