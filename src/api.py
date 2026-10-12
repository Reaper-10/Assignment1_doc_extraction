from pathlib import Path
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.document_reader import read_document
from src.metadata_extractor import extract_metadata

app = FastAPI(title="Rental Agreement Metadata Extractor", version="1.0.0")
ALLOWED = {".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, "Upload a DOCX or image file.")
    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(413, "File is too large (maximum 15 MB).")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            temp_path = Path(tmp.name)
        text = read_document(temp_path)
        result = extract_metadata(text)
        return {"filename": file.filename, "metadata": result}
    except Exception as exc:
        raise HTTPException(500, f"Extraction failed: {type(exc).__name__}: {exc}")
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
