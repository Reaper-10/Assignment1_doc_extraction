from pathlib import Path
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.document_reader import read_document
from src.metadata_extractor import extract_metadata

app = FastAPI(title="Rental Agreement Metadata Extractor", version="1.1.0")
ALLOWED = {".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_BYTES = 15 * 1024 * 1024

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=415, detail="Upload a DOCX or supported image file.")

    content = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="File is too large (maximum 15 MB).")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            temp_path = Path(tmp.name)
        text = read_document(temp_path)
        if not text.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted from the uploaded document.")
        result = extract_metadata(text)
        return {"filename": file.filename, "metadata": result}
    except HTTPException:
        raise
    except Exception:
        # Avoid returning internal exception details to the API caller.
        raise HTTPException(status_code=500, detail="Extraction failed. Check server logs for details.")
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
        await file.close()
