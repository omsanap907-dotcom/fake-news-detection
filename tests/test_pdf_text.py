from io import BytesIO

import pytest
from pypdf import PdfWriter

from fake_news.pdf_text import PdfTextError, extract_pdf_text


def make_pdf_with_text(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    # pypdf does not provide a high-level text drawing API; use a minimal blank PDF
    # here to test page extraction failure paths.
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_rejects_empty_bytes():
    with pytest.raises(PdfTextError, match="empty"):
        extract_pdf_text(b"")


def test_rejects_non_pdf_bytes():
    with pytest.raises(PdfTextError, match="does not appear"):
        extract_pdf_text(b"not a pdf")


def test_rejects_blank_pdf_without_text():
    pdf = make_pdf_with_text("some article text")
    with pytest.raises(PdfTextError, match="No selectable text"):
        extract_pdf_text(pdf)
