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
    # CSV file names may omit the extension or include a secondary extension such as .pdf.docx.
    direct = folder / name
    if direct.exists():
        return direct

    normalized = normalize_filename(name)
    matches = []
    for path in folder.iterdir():
        if path.is_file():
            if path.name == name or path.stem == Path(name).stem or normalize_filename(path.name) == normalized:
                matches.append(path)
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
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    input_dir = data_dir / args.split
    labels_path = data_dir / f"{args.split}.csv"
    if not input_dir.is_dir() or not labels_path.exists():
        raise FileNotFoundError(
            f"Expected {input_dir} and {labels_path}. Extract the supplied dataset into --data-dir."
        )
    df = pd.read_csv(labels_path)
    if "File Name" not in df:
        raise ValueError("CSV needs a 'File Name' column.")
    work = df if args.limit is None else df.head(args.limit)
    rows = []
    for i, record in work.iterrows():
        filename = str(record["File Name"])
        path = resolve_file(filename, input_dir)
        print(f"[{len(rows)+1}/{len(work)}] Reading {path.name}")
        text = read_document(path)
        if not text:
            print("  WARNING: no text extracted; model will return nulls.")
        values = extract_metadata(text)
        rows.append({"File Name": filename, **{field: values.get(field) for field in FIELDS}})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_predictions.csv"
    pd.DataFrame(rows, columns=["File Name", *FIELDS]).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} predictions to {output_path}")

if __name__ == "__main__":
    main()
