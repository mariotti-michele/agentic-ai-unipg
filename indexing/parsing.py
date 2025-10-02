import json
import camelot
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path

from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document

from scraping import sha

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="camelot")

import subprocess
import tempfile
from pathlib import Path

def to_documents_from_html(file_path: Path, source_url: str, page_title: str) -> list[Document]:
    raw_html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")

    main_el = soup.find("main")
    if not main_el:
        print(f"[WARN] Nessun <main> trovato in {source_url}, salto")
        return []
    main_html = str(main_el)
    tmp_path = file_path.with_suffix(".main.html")
    tmp_path.write_text(main_html, encoding="utf-8")

    elements = partition_html(filename=str(tmp_path), include_page_breaks=False, languages=["ita", "eng"])
    docs = []
    crawl_ts = datetime.now(timezone.utc).isoformat()

    for el in elements:
        text = getattr(el, "text", None) or str(el).strip()
        if not text:
            continue

        element_type = getattr(el, "category", None) or el.__class__.__name__
        meta = getattr(el, "metadata", None)
        meta = meta.to_dict() if meta is not None else {}

        doc = Document(
            page_content=text,
            metadata={
                "source_url": source_url,
                "doc_type": "html",
                "page_title": page_title,
                "element_type": element_type,
                "lang": meta.get("languages", None),
                "crawl_ts": crawl_ts,
                "doc_id": sha(source_url),
            },
        )
        docs.append(doc)
    return docs


def to_documents_from_pdf(file_path: Path, source_url: str) -> list[Document]:
    docs = []
    crawl_ts = datetime.now(timezone.utc).isoformat()

    strategies = ["hi_res", "auto", "ocr_only"]
    elements = []
    for strat in strategies:
        try:
            elements = partition_pdf(
                filename=str(file_path),
                strategy=strat,
                include_page_breaks=True,
                infer_table_structure=True,
                languages=["ita", "eng"]
            )
            if elements: 
                print(f"[INFO] Estratti elementi da {file_path.name} con strategy={strat}")
                break
        except Exception as e:
            print(f"[WARN] partition_pdf fallito con strategy={strat} su {file_path}: {e}")

    if not elements:
        print(f"[WARN] Nessun contenuto estratto da {file_path}, provo fallback OCR manuale...")
        try:
            from parsing import parse_pdf_with_ocr
            ocr_text = parse_pdf_with_ocr(str(file_path), lang="ita+eng")
            if ocr_text.strip():
                elements = [type("FakeElement", (), {"text": ocr_text, "category": "OCRText", "metadata": {}})()]
                print(f"[INFO] Estratto testo via fallback OCR da {file_path.name}")
            else:
                print(f"[WARN] OCR manuale non ha estratto testo da {file_path.name}")
                return []
        except Exception as e:
            print(f"[ERROR] OCR manuale fallito su {file_path}: {e}")
            return []

    for el in elements:
        text = getattr(el, "text", None) or str(el).strip()
        if not text:
            continue
        element_type = getattr(el, "category", None) or el.__class__.__name__
        meta = getattr(el, "metadata", None)
        meta = meta.to_dict() if meta is not None else {}

        doc = Document(
            page_content=text,
            metadata={
                "source_url": source_url,
                "doc_type": "pdf",
                "page_number": meta.get("page_number", None),
                "file_name": file_path.name,
                "element_type": element_type,
                "lang": meta.get("languages", None),
                "crawl_ts": crawl_ts,
                "doc_id": sha(source_url),
            },
        )
        docs.append(doc)

    tables = extract_tables_camelot(file_path)
    for t in tables:
        docs.append(Document(
            page_content=json.dumps(t["row"], ensure_ascii=False),
            metadata={
                "source_url": source_url,
                "doc_type": "pdf-table",
                "page_number": t["page_number"],
                "file_name": t["file_name"],
                "table_index": t["table_index"],
                "element_type": "Table",
                "crawl_ts": crawl_ts,
                "doc_id": sha(source_url)
            }
        ))
    return docs

def extract_tables_camelot(file_path: Path) -> list[dict]:
    records = []
    try:
        tables = camelot.read_pdf(str(file_path), pages="all", flavor="lattice")
        if not tables or len(tables) == 0:
            tables = camelot.read_pdf(str(file_path), pages="all", flavor="stream")
        for t_idx, table in enumerate(tables):
            df = table.df
            for i in range(len(df)):
                row = df.iloc[i].to_dict()
                records.append({
                    "row": row,
                    "page_number": table.page,
                    "file_name": file_path.name,
                    "table_index": t_idx
                })
    except Exception as e:
        print(f"[WARN] Camelot fallito su {file_path}: {e}")
    return records

def parse_pdf_with_ocr(pdf_path: str, lang: str = "ita+eng") -> str:
    """
    Esegue OCR su un PDF immagine (es. orari, calendari) usando pdftoppm + tesseract.
    
    Args:
        pdf_path (str): percorso al file PDF
        lang (str): lingua OCR ("ita", "eng", "ita+eng", ...)

    Returns:
        str: testo estratto
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File non trovato: {pdf_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        subprocess.run(
            ["pdftoppm", "-png", str(pdf_path), str(tmpdir / "page")],
            check=True
        )

        text_chunks = []
        for img_file in sorted(tmpdir.glob("page-*.png")):
            output_txt = tmpdir / "out"
            subprocess.run(
                ["tesseract", str(img_file), str(output_txt), "-l", lang],
                check=True
            )
            with open(str(output_txt) + ".txt", "r", encoding="utf-8") as f:
                text_chunks.append(f.read())

        return "\n".join(text_chunks)