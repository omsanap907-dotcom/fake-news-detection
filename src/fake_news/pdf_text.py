"""Extract selectable text from a PDF for downstream analysis.

This module intentionally does not classify claims. Scanned PDFs without an OCR
layer return an informative error; OCR can be added separately later.
"""
from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader


class PdfTextError(ValueError):
    """Raised when a PDF cannot be read or has no extractable text."""


def extract_pdf_text(data: bytes, *, max_pages: int = 40, max_chars: int = 200_000) -> dict[str, object]:
    """Extract text from an uploaded PDF byte string with basic size limits."""
    if not isinstance(data, bytes) or not data:
        raise PdfTextError("The uploaded file is empty.")
    if not data.startswith(b"%PDF-"):
        raise PdfTextError("The uploaded file does not appear to be a PDF.")
    try:
        reader = PdfReader(BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise PdfTextError("Password-protected PDFs are not supported.")
        page_count = len(reader.pages)
        if page_count == 0:
            raise PdfTextError("The PDF contains no pages.")
        if page_count > max_pages:
            raise PdfTextError(f"The PDF has {page_count} pages; the limit is {max_pages}.")
        chunks: list[str] = []
        total = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            remaining = max_chars - total
            if remaining <= 0:
                break
            page_text = page_text[:remaining]
            chunks.append(page_text)
            total += len(page_text)
        text = "\n\n".join(chunks).strip()
        if not text:
            raise PdfTextError("No selectable text was found. This may be a scanned PDF; OCR is not enabled yet.")
        return {"text": text, "page_count": page_count, "character_count": len(text)}
    except PdfTextError:
        raise
    except Exception as exc:
        raise PdfTextError("Could not read this PDF. Check that it is a valid, uncorrupted document.") from exc
