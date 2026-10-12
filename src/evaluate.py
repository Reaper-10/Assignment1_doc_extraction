from pathlib import Path
from decimal import Decimal, InvalidOperation
import pandas as pd

FIELD_MAP = {
    "Agreement Value": "Aggrement Value",
    "Agreement Start Date": "Aggrement Start Date",
    "Agreement End Date": "Aggrement End Date",
    "Renewal Notice (Days)": "Renewal Notice (Days)",
    "Party One": "Party One",
    "Party Two": "Party Two",
}
NUMERIC_FIELDS = {"Agreement Value", "Renewal Notice (Days)"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def clean_value(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())

def exact_match(field, actual, predicted):
    a, b = clean_value(actual), clean_value(predicted)
    if field in NUMERIC_FIELDS:
        if not a or not b:
            return a == b
        try:
            # Treat equivalent numeric representations (e.g., 6500 and 6500.0) as equal.
            return Decimal(a.replace(",", "")) == Decimal(b.replace(",", ""))
        except InvalidOperation:
            return a == b
    # Preserve case and punctuation for names and dates; only whitespace is normalized.
    return a == b

def evaluate(
    predictions_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    output_dir: str | Path | None = None,
):
    predictions_path = Path(predictions_path) if predictions_path else PROJECT_ROOT / "outputs" / "test_predictions.csv"
    labels_path = Path(labels_path) if labels_path else PROJECT_ROOT / "data" / "test.csv"
    output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs"

    preds = pd.read_csv(predictions_path, dtype=str).fillna("")
    labels = pd.read_csv(labels_path, dtype=str).fillna("")
    if "File Name" not in preds or "File Name" not in labels:
        raise ValueError("Both CSVs must contain a 'File Name' column.")

    label_cols = {col: f"{col}_true" for col in labels.columns if col != "File Name"}
    pred_cols = {col: f"{col}_pred" for col in preds.columns if col != "File Name"}
    labels2 = labels.rename(columns=label_cols)
    preds2 = preds.rename(columns=pred_cols)
    merged = labels2.merge(preds2, on="File Name", validate="one_to_one")

    rows = []
    for output_field, label_col in FIELD_MAP.items():
        true_col = f"{label_col}_true"
        pred_col = f"{output_field}_pred"
        if true_col not in merged or pred_col not in merged:
            raise ValueError(f"Missing field in CSV: {true_col} or {pred_col}")
        matches = [exact_match(output_field, a, b) for a, b in zip(merged[true_col], merged[pred_col])]
        rows.append({
            "Field": output_field,
            "Exact Matches": int(sum(matches)),
            "Documents": int(len(matches)),
            "Recall": (sum(matches) / len(matches)) if matches else 0.0,
            "Recall (%)": 100 * (sum(matches) / len(matches)) if matches else 0.0,
        })
    report = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "per_field_recall.csv", index=False)
    print(report.to_string(index=False, formatters={"Recall": "{:.3f}".format, "Recall (%)": "{:.1f}".format}))
    print("\\nComparison: numeric fields compare numerically; dates/names use exact string equality after whitespace normalization.")
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions")
    parser.add_argument("--labels")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    evaluate(args.predictions, args.labels, args.output_dir)
