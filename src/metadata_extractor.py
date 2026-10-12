import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors

FIELDS = [
    "Agreement Value",
    "Agreement Start Date",
    "Agreement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
]

SYSTEM_PROMPT = """You extract metadata from rental agreements.
Interpret the full document semantically. The document layout and wording may vary.
Return only values supported by the document. Do not invent missing values.
Distinguish rent/agreement value from deposits, fees, and unrelated amounts.
Distinguish agreement start/end dates from signing dates.
Identify the two contracting parties, not witnesses, agents, or guarantors.
For Renewal Notice (Days), extract the notice period relevant to renewal/termination
that best matches the dataset labels. If the document is ambiguous, return null.
Preserve names and the date format used in the document when possible.
Return valid JSON with exactly these keys:
Agreement Value, Agreement Start Date, Agreement End Date,
Renewal Notice (Days), Party One, Party Two.
Numeric fields should be numbers when clear; unknown fields should be null."""


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def _normalize_field_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"null", "none", "n/a", "na"}:
            return None
    if field in {"Agreement Value", "Renewal Notice (Days)"}:
        if isinstance(value, str):
            compact = value.replace(",", "").replace("Rs", "").strip()
            if compact.endswith("/-"):
                compact = compact[:-2]
            if compact:
                try:
                    return float(compact)
                except ValueError:
                    return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        return value
    return value


def _normalize_result(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in FIELDS:
        if field not in payload:
            normalized[field] = None
        else:
            normalized[field] = _normalize_field_value(field, payload.get(field))
    return normalized


def _call_gemini(document_text: str, model: str) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=api_key)
    user_prompt = (
        "Extract the six fields from the rental agreement below. "
        "Do not use assumptions beyond the supplied text. Return only a JSON object "
        "with exactly the requested keys; represent unknown values as null.\n\n"
        "<DOCUMENT>\n" + document_text[:100000] + "\n</DOCUMENT>"
    )
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "temperature": 0,
                },
            )
            break
        except errors.ServerError as exc:
            if exc.code != 503 or attempt == 2:
                raise
            delay = 5 * (2**attempt)
            print(f"Gemini is temporarily unavailable; retrying in {delay} seconds.")
            time.sleep(delay)

    raw = response.text
    if not raw:
        raise ValueError("Gemini returned an empty response.")
    result = _extract_json_object(raw)
    return _normalize_result(result)


def extract_metadata(document_text: str, model: str | None = None) -> dict[str, Any]:
    if not document_text.strip():
        return {field: None for field in FIELDS}

    load_dotenv()
    model = model or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
    return _call_gemini(document_text, model)
