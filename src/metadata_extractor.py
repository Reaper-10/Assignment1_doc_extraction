import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
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
NUMERIC_FIELDS = {"Agreement Value", "Renewal Notice (Days)"}
DATE_FIELDS = {"Agreement Start Date", "Agreement End Date"}

SYSTEM_PROMPT = """You extract metadata from rental agreements.

Use only evidence in the supplied document. Never invent a value.
Return one JSON object with exactly the six requested keys.

Rules:
1. Agreement Value: extract the agreed rent/value, not a deposit, fee,
   penalty, or unrelated monetary amount.
2. Agreement Start Date and Agreement End Date: Extract the contractual start and end dates of the agreement.
   If the document explicitly states the end date, use it.
   If the agreement gives a start date and an explicit lease duration but no end date, determine the
   contractual expiry date by reasoning from the stated duration and the agreement's date convention.
   For a term expressed in months, distinguish the final day of the contractual term from the
   anniversary date marking the next period. Do not use the signing date as the start date unless
   the agreement explicitly defines it that way. If the end date cannot be determined reliably, return null.
3. Renewal Notice (Days): extract the applicable renewal/termination
   notice period in days. Do not confuse it with the lease duration.
   If absent or ambiguous, return null.
4. Party One and Party Two: Extract the complete contracting-party names exactly as written in the agreement.
   Preserve prefixes such as Sri, Mr., or Mrs., initials, punctuation, and company suffixes.
   Do not add or remove parts of a name to match assumptions. Return plain strings, not dictionaries.
   Identify the lessor and lessee correctly and preserve the order expected by the assignment.
5. Output types: numeric fields are JSON numbers when clear; dates and
   party names are strings; unknown values are null. Do not include
   explanations, markdown, or additional keys.
"""

def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Gemini response did not contain a JSON object.")
    payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Gemini response JSON must be an object.")
    return payload

def _normalize_date(value: str) -> str:
    value = value.strip()
    # Normalize common date representations without calendar-validating them:
    # company-provided labels may contain source anomalies that must be preserved.
    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", value)
    if match:
        year, month, day = match.groups()
        return f"{int(day):02d}.{int(month):02d}.{year}"
    match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", value)
    if match:
        day, month, year = match.groups()
        return f"{int(day):02d}.{int(month):02d}.{year}"
    return value

def _normalize_field_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"null", "none", "n/a", "na", "unknown"}:
            return None

    if field in NUMERIC_FIELDS:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            compact = value.replace(",", "").strip()
            compact = re.sub(r"^(?:Rs\.?|INR|₹)\s*", "", compact, flags=re.IGNORECASE)
            compact = compact.removesuffix("/-").strip()
            try:
                number = Decimal(compact)
                if not number.is_finite():
                    return None
                return int(number) if number == number.to_integral_value() else float(number)
            except (InvalidOperation, ValueError):
                return None
        return None

    if field in DATE_FIELDS:
        if not isinstance(value, str):
            return None
        return _normalize_date(value)

    if field in {"Party One", "Party Two"}:
        if isinstance(value, str):
            return value.strip() or None
        # Be defensive if the model violates the requested string schema.
        # Only accept a clear name property; do not stringify an arbitrary object.
        if isinstance(value, dict):
            for key in ("name", "party_name", "Party Name"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        return None

    return value

def _normalize_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {field: _normalize_field_value(field, payload.get(field)) for field in FIELDS}

def _call_gemini(document_text: str, model: str) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    user_prompt = (
        "Extract the six fields from the rental agreement below. "
        "Use only the document text. Return exactly the requested JSON keys. "
        "Party One and Party Two must be plain strings or null.\n\n"
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
            if getattr(exc, "code", None) != 503 or attempt == 2:
                raise
            delay = 5 * (2 ** attempt)
            print(f"Gemini is temporarily unavailable; retrying in {delay} seconds.")
            time.sleep(delay)

    raw = response.text
    if not raw:
        raise ValueError("Gemini returned an empty response.")
    return _normalize_result(_extract_json_object(raw))

def extract_metadata(document_text: str, model: str | None = None) -> dict[str, Any]:
    if not isinstance(document_text, str) or not document_text.strip():
        return {field: None for field in FIELDS}
    load_dotenv()
    model = model or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
    return _call_gemini(document_text, model)
