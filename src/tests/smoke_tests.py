import asyncio
from pathlib import Path
import os

from src.services.ocr_service import ocr_service
from src.services.text_processor import text_processor
from src.services.embedding_service import embedding_service
from src.config.config_settings import settings


async def test_ocr(pdf_path: str):
    # Throttle OCR for stability in tests
    os.environ.setdefault("DOTS_OCR_MAX_NEW_TOKENS", "256")
    settings.DOTS_OCR_DPI = 150
    settings.DOTS_OCR_MAX_IMAGE_WIDTH = 960
    result = await ocr_service.extract_text_from_pdf(pdf_path, max_pages=1)
    assert result.text and result.page_count > 0
    print("OCR OK:", result.page_count, "pages")


async def test_ocr_plus_chunking(pdf_path: str):
    ocr = await ocr_service.extract_text_from_pdf(pdf_path, max_pages=1)
    chunks = await text_processor.process_and_chunk_text(ocr.text, pdf_path, metadata={})
    assert chunks and len(chunks) > 0
    print("Chunking OK:", len(chunks), "chunks")


async def test_full_embedding(pdf_path: str):
    ocr = await ocr_service.extract_text_from_pdf(pdf_path, max_pages=1)
    chunks = await text_processor.process_and_chunk_text(ocr.text, pdf_path, metadata={})
    embs = await embedding_service.generate_embeddings(chunks)
    assert embs and len(embs) == len(chunks)
    dim = len(embs[0].embedding) if embs else 0
    print("Embeddings OK:", len(embs), "dim=", dim)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.tests.smoke_tests <pdf_path>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    asyncio.run(test_ocr(pdf_path))
    asyncio.run(test_ocr_plus_chunking(pdf_path))
    asyncio.run(test_full_embedding(pdf_path))


