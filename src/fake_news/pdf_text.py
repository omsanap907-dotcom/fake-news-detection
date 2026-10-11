"""Bounded extraction of selectable text from uploaded PDF bytes."""
from __future__ import annotations
from io import BytesIO
from pypdf import PdfReader

class PdfTextError(ValueError):
    """A user-correctable PDF upload problem."""

def extract_pdf_text(data: bytes, *, max_pages: int = 40, max_chars: int = 200_000) -> dict[str, object]:
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
        chunks, total = [], 0
        for page in reader.pages:
            remaining = max_chars - total
            if remaining <= 0:
                break
            text = (page.extract_text() or "")[:remaining]
            chunks.append(text)
            total += len(text)
        extracted = "\n\n".join(chunks).strip()
        if not extracted:
            raise PdfTextError("No selectable text was found. This may be a scanned PDF; OCR is not enabled yet.")
        return {"text": extracted, "page_count": page_count, "character_count": len(extracted)}
    except PdfTextError:
        raise
    except Exception as exc:
        raise PdfTextError("Could not read this PDF. Check that it is valid and uncorrupted.") from exc
