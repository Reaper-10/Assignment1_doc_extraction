# Rental Agreement Metadata Extraction

An AI-powered document extraction project that reads rental agreements in DOCX or scanned-image format and identifies six key metadata fields. It uses OCR to read images and Google Gemini to interpret document content semantically, without relying on rule-based field extraction.

## Key Features

- Extracts **Agreement Value**, **Agreement Start Date**, **Agreement End Date**, **Renewal Notice (Days)**, **Party One**, and **Party Two**.
- Reads DOCX paragraphs and tables, and OCRs PNG, JPG/JPEG, TIFF, and BMP images.
- Uses Gemini with JSON output and a configurable model (`gemini-3.8-flash` by default).
- Retries temporary Gemini `503 UNAVAILABLE` errors up to two additional times with increasing delays; other errors are reported.
- Generates CSV predictions for the training or test split and supports processing a limited number of files for a smoke test.
- Reports per-field exact-match recall against labelled CSV data.
- Provides an optional FastAPI endpoint for document uploads.

## Tech Stack

- **Frontend:** None; the project provides a command-line interface and an optional interactive API page.
- **Backend:** Python, FastAPI, Uvicorn, Google Gemini API (`google-genai`).
- **Database:** None; inputs, predictions, and evaluation reports are stored as files.
- **Document processing:** `python-docx`, Pillow, Tesseract OCR, `pytesseract`.
- **Data and configuration:** pandas, `python-dotenv`, CSV, environment variables.

## Prerequisites

- Python 3.10 or newer.
- A Google Gemini API key, with access to the selected Gemini model.
- Tesseract OCR installed and available on `PATH` for scanned-image input. DOCX extraction does not require Tesseract.
- The assignment dataset, organized as described in [Dataset Layout](#dataset-layout).
- An internet connection for Gemini API requests.

## Installation & Setup

Run commands from the project root directory.

### 1. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure Gemini

Edit the single project-root `.env` file and set `GEMINI_API_KEY` to your Google Gemini API key. Keep the existing project settings in this same file:

```dotenv
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3.8-flash
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

`GEMINI_MODEL` is optional; if omitted, the application defaults to `gemini-3.8-flash`. `TESSERACT_CMD` is optional and only needed if Tesseract is not in a standard location or on `PATH`. Keep `.env` private and never commit API keys.

### 4. Install and configure Tesseract OCR

Install Tesseract for your operating system and verify it is callable:

```text
tesseract --version
```

If Tesseract is installed outside the locations automatically checked by the project, set `TESSERACT_CMD` in the same `.env` file to the full path of its executable.

### 5. Add the assignment dataset

Extract the supplied assignment dataset into the project so it has the following layout:

### Dataset Layout

```text
data/
├── train.csv
├── test.csv
├── train/
│   └── agreement documents (.docx and scanned images)
└── test/
    └── agreement documents (.docx and scanned images)
```

The CSV files must contain a `File Name` column and metadata columns for the six target fields. The supplied labels spell Agreement as `Aggrement Value` and `Aggrement Start Date`; the evaluator maps these columns to the prediction field names.

## Run Locally

### Generate predictions

From the project root, run a one-file smoke test:

```bash
python main.py --split test --limit 1
```

Generate predictions for the full test set:

```bash
python main.py --split test
```

Generate training-set predictions for error analysis:

```bash
python main.py --split train
```

Predictions are written to `outputs/test_predictions.csv` or `outputs/train_predictions.csv`, depending on the selected split. Use `--data-dir` and `--output-dir` to specify different dataset or output folders:

```bash
python main.py --split test --data-dir data --output-dir outputs
```

The CLI reads configuration from `.env`. Each non-empty document is sent to Gemini for extraction; the document text is limited to 100,000 characters per request. Gemini API usage may be subject to account quotas and charges.

### Evaluate per-field recall

After generating predictions, compare them with the corresponding labelled CSV:

```bash
python -m src.evaluate --predictions outputs/test_predictions.csv --labels data/test.csv
```

The default inputs and report directory are resolved from the project root, so from inside the `src` folder you can also run:

```bash
python evaluate.py
```

The evaluator prints the exact-match recall for each of the six fields and saves the report to `outputs/per_field_recall.csv`. Numeric fields are compared numerically (for example, `6500` and `6500.0` match); dates and party names use exact string comparison after whitespace normalization. The score depends on the generated predictions; run the evaluation to produce current metrics rather than relying on stale output files.

### Latest verified test run

The four-document test split was run on 2026-10-13 with `gemini-3.1-flash-lite` selected through `GEMINI_MODEL`. All four documents completed without API errors. The recorded evaluation is **18/24 exact field matches (75.0%)**:

| Field | Exact matches |
| --- | ---: |
| Agreement Value | 4/4 (100%) |
| Agreement Start Date | 4/4 (100%) |
| Agreement End Date | 2/4 (50%) |
| Renewal Notice (Days) | 3/4 (75%) |
| Party One | 4/4 (100%) |
| Party Two | 1/4 (25%) |

To use the same model in Windows PowerShell for a run, set the model in the current shell before running the commands:

```powershell
$env:GEMINI_MODEL = "gemini-3.1-flash-lite"
python main.py --split test
python -m src.evaluate --predictions outputs/test_predictions.csv --labels data/test.csv
```

The score measures agreement with the supplied labels, not guaranteed factual correctness. Review predictions against the source document where labels, date conventions, or party-name formatting are ambiguous. Free-tier quota and model availability vary; `429 RESOURCE_EXHAUSTED` and `503 UNAVAILABLE` errors can still occur. The CLI records failed documents in `outputs/test_run_errors.csv`; do not interpret an incomplete run's evaluation as a completed result.

## Optional REST API

Start the FastAPI service from the project root:

```bash
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs` to try `POST /extract`. Upload a DOCX or supported image file; the endpoint returns the original filename and extracted metadata. Uploads are limited to 15 MB.

The endpoint can also be called with `curl`:

```bash
curl.exe -X POST "http://127.0.0.1:8000/extract" -F "file=@data/test/your-agreement.png"
```

The API uses the same `GEMINI_API_KEY` and optional `GEMINI_MODEL` settings as the CLI.

## Extraction Approach & Notes

1. DOCX documents are read from paragraphs and tables. Scanned images are preprocessed and passed through Tesseract OCR.
2. The extracted text is sent to Gemini with instructions to identify the requested fields, avoid unsupported assumptions, and return unknown values as `null`.
3. Numeric predictions are normalized when possible, then written as CSV or returned by the API.

OCR is used only to obtain text; field identification is performed semantically by the language model, not with regular-expression or static template rules. The supplied labelled sample is small, so the project uses a pretrained model without fine-tuning. Agreement terms such as renewal notice periods may be ambiguous, and OCR quality can affect extracted values. Review predictions and OCR text where needed. Do not tune prompts against test labels.
