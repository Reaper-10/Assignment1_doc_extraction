from pathlib import Path
import calendar
from decimal import Decimal, InvalidOperation
import re
import pandas as pd

FIELD_MAP = {
    "Agreement Value": "Aggrement Value",  # Preserve the company CSV's original header spelling.
    "Agreement Start Date": "Aggrement Start Date",
    "Agreement End Date": "Aggrement End Date",
    "Renewal Notice (Days)": "Renewal Notice (Days)",
    "Party One": "Party One",
    "Party Two": "Party Two",
}
NUMERIC_FIELDS = {"Agreement Value", "Renewal Notice (Days)"}
DATE_FIELDS = {"Agreement Start Date", "Agreement End Date"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MONTHS = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}

def clean_value(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())

def normalize_date(value) -> str:
    """Canonicalize common date formats without changing invalid source dates."""
    value = clean_value(value)
    if not value:
        return ""
    # ISO yyyy-mm-dd
    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", value)
    if match:
        year, month, day = match.groups()
        return f"{int(day):02d}.{int(month):02d}.{year}"
    # Numeric day/month/year formats
    match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", value)
    if match:
        day, month, year = match.groups()
        return f"{int(day):02d}.{int(month):02d}.{year}"
    month_pattern = "|".join(MONTHS)
    match = re.fullmatch(
        rf"(\d{{1,2}})(?:st|nd|rd|th|\*)?\s+(?:of\s+)?"
        rf"({month_pattern}),?\s+(\d{{4}})",
        value, flags=re.IGNORECASE,
    )
    if match:
        day, month_name, year = match.groups()
        return f"{int(day):02d}.{MONTHS[month_name.lower()]:02d}.{year}"
    match = re.fullmatch(rf"end of ({month_pattern})\s+(\d{{4}})", value, flags=re.IGNORECASE)
    if match:
        month_name, year = match.groups()
        last_day = calendar.monthrange(int(year), MONTHS[month_name.lower()])[1]
        return f"{last_day:02d}.{MONTHS[month_name.lower()]:02d}.{year}"
    return value

def exact_match(field: str, actual, predicted) -> bool:
    a, b = clean_value(actual), clean_value(predicted)
    if field in DATE_FIELDS:
        return normalize_date(a) == normalize_date(b)
    if field in NUMERIC_FIELDS:
        if not a or not b:
            return a == b
        try:
            return Decimal(a.replace(",", "")) == Decimal(b.replace(",", ""))
        except InvalidOperation:
            return a == b
    # Party names remain strict: do not remove titles or rewrite names automatically.
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
    for name, frame in (("predictions", preds), ("labels", labels)):
        if "File Name" not in frame.columns:
            raise ValueError(f"{name.title()} CSV must contain a 'File Name' column.")
        if frame["File Name"].duplicated().any():
            duplicates = frame.loc[frame["File Name"].duplicated(), "File Name"].tolist()
            raise ValueError(f"Duplicate File Name values in {name}: {duplicates}")
    missing_predictions = sorted(set(labels["File Name"]) - set(preds["File Name"]))
    unexpected_predictions = sorted(set(preds["File Name"]) - set(labels["File Name"]))
    if missing_predictions or unexpected_predictions:
        raise ValueError(
            "Prediction coverage does not match labels. "
            f"Missing predictions: {missing_predictions}; "
            f"unexpected predictions: {unexpected_predictions}. "
            "Check the run before interpreting the score."
        )

    label_cols = {col: f"{col}_true" for col in labels.columns if col != "File Name"}
    pred_cols = {col: f"{col}_pred" for col in preds.columns if col != "File Name"}
    merged = labels.rename(columns=label_cols).merge(
        preds.rename(columns=pred_cols), on="File Name", how="left", validate="one_to_one"
    )

    rows = []
    total_matches = 0
    total_fields = len(merged) * len(FIELD_MAP)
    for output_field, label_col in FIELD_MAP.items():
        true_col, pred_col = f"{label_col}_true", f"{output_field}_pred"
        if true_col not in merged.columns or pred_col not in merged.columns:
            raise ValueError(f"Missing field in CSV: {true_col} or {pred_col}")
        matches = [exact_match(output_field, a, b) for a, b in zip(merged[true_col], merged[pred_col])]
        count = int(sum(matches))
        total_matches += count
        docs = len(matches)
        rows.append({
            "Field": output_field,
            "Exact Matches": count,
            "Documents": docs,
            "Exact Match Rate": count / docs if docs else 0.0,
            "Exact Match Rate (%)": 100 * count / docs if docs else 0.0,
        })

    report = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "per_field_recall.csv", index=False)
    report.to_csv(output_dir / "per_field_exact_match.csv", index=False)
    print(report.to_string(index=False, formatters={
        "Exact Match Rate": "{:.3f}".format,
        "Exact Match Rate (%)": "{:.1f}".format,
    }))
    overall = total_matches / total_fields if total_fields else 0.0
    print(f"\nOverall exact field-match rate: {total_matches}/{total_fields} = {overall:.2%}")
    print("Dates are format-normalized; numeric fields compare numerically; party names use strict string equality.")
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions")
    parser.add_argument("--labels")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    evaluate(args.predictions, args.labels, args.output_dir)
