from pathlib import Path
import argparse
import os
import pandas as pd
from dotenv import load_dotenv
from src.document_reader import read_document
from src.metadata_extractor import extract_metadata, FIELDS

load_dotenv()

def normalize_filename(name: str) -> str:
    candidate = Path(name).name
    while True:
        new_candidate = Path(candidate).stem
        if new_candidate == candidate:
            return candidate
        candidate = new_candidate

def resolve_file(name: str, folder: Path) -> Path:
    direct = folder / name
    if direct.exists():
        return direct
    normalized = normalize_filename(name)
    matches = [
        path for path in folder.iterdir()
        if path.is_file() and (
            path.name == name
            or path.stem == Path(name).stem
            or normalize_filename(path.name) == normalized
        )
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No file found for CSV entry {name!r} in {folder}")
    raise RuntimeError(f"Ambiguous file match for {name!r}: {matches}")

def main():
    parser = argparse.ArgumentParser(description="Extract metadata from rental agreements.")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"))
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "outputs"))
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit; omit for the full split.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    input_dir = data_dir / args.split
    labels_path = data_dir / f"{args.split}.csv"
    if not input_dir.is_dir() or not labels_path.exists():
        raise FileNotFoundError(
            f"Expected documents folder {input_dir} and CSV {labels_path}. "
            "Check --data-dir and the supplied dataset layout."
        )
    df = pd.read_csv(labels_path, dtype=str).fillna("")
    if "File Name" not in df.columns:
        raise ValueError("CSV needs a 'File Name' column.")
    if df["File Name"].duplicated().any():
        raise ValueError("Input CSV contains duplicate File Name values; resolve the ambiguity without editing the source file.")

    work = df if args.limit is None else df.head(args.limit)
    rows, errors = [], []
    for index, record in work.iterrows():
        filename = str(record["File Name"])
        print(f"[{len(rows)+1}/{len(work)}] {filename}")
        try:
            path = resolve_file(filename, input_dir)
            print(f"  Reading {path.name}")
            text = read_document(path)
            if not text.strip():
                raise ValueError("No text extracted from document; inspect the source file/OCR setup.")
            values = extract_metadata(text)
            row = {"File Name": filename, **{field: values.get(field) for field in FIELDS}}
            rows.append(row)
            print("  Extraction completed.")
        except Exception as exc:
            # Keep one output row per input row so failures cannot silently disappear.
            rows.append({"File Name": filename, **{field: None for field in FIELDS}})
            errors.append({
                "CSV Row": int(index) + 2,
                "File Name": filename,
                "Error Type": type(exc).__name__,
                "Error": str(exc),
            })
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_predictions.csv"
    pd.DataFrame(rows, columns=["File Name", *FIELDS]).to_csv(output_path, index=False)
    error_path = output_dir / f"{args.split}_run_errors.csv"
    pd.DataFrame(errors, columns=["CSV Row", "File Name", "Error Type", "Error"]).to_csv(error_path, index=False)
    print(f"\nSaved {len(rows)} predictions to {output_path}")
    print(f"Run errors: {len(errors)}; details saved to {error_path}")
    if args.limit is not None:
        print("NOTE: --limit was supplied. This is a partial run and must not be used as the full-split evaluation.")

if __name__ == "__main__":
    main()
