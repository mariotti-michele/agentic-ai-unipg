import json
import camelot
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path

from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from scraping import sha

import logging, warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="camelot")


# === Funzione comune per semantic chunking ===
def semantic_chunk(elements, source_url, page_title, doc_type, file_name=None):
    from unstructured.documents.elements import Title, Header, NarrativeText, ListItem, Text, Table

    crawl_ts = datetime.now(timezone.utc).isoformat()
    docs, merged_chunks = [], []
    current_chunk, current_header = "", None

    for el in elements:
        text = getattr(el, "text", None)
        if not text:
            continue
        text = text.replace("\n", " ").strip()
        if not text:
            continue


        if isinstance(el, (Title, Header)):
            if current_chunk.strip():
                merged_chunks.append(current_chunk.strip())
                current_chunk = ""
            current_header = text.strip()
            continue


        if isinstance(el, (NarrativeText, ListItem, Text)):
            if current_header:
                current_chunk = f"{current_header}\n{text}"
                current_header = None
            else:
                current_chunk += ("\n" if current_chunk else "") + text
            continue


        if isinstance(el, Table):
            if current_chunk.strip():
                merged_chunks.append(current_chunk.strip())
                current_chunk = ""
            merged_chunks.append(f"[TABELLA]\n{text}")


    if current_chunk.strip():
        merged_chunks.append(current_chunk.strip())


    for idx, chunk in enumerate(merged_chunks):
        docs.append(Document(
            page_content=chunk,
            metadata={
                "source_url": source_url,
                "doc_type": doc_type,
                "page_title": page_title,
                "file_name": file_name,
                "element_type": "SemanticChunk",
                "lang": ["ita"],
                "crawl_ts": crawl_ts,
                "doc_id": sha(f"{source_url}_{idx}"),
            },
        ))

    print(f"[INFO] {len(merged_chunks)} chunk semantici creati da {source_url}")
    return docs


# === HTML ===
def to_documents_from_html(file_path: Path, source_url: str, page_title: str) -> list[Document]:
    raw_html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")

    main_el = soup.find("main")
    if not main_el:
        print(f"[WARN] Nessun <main> trovato in {source_url}, salto")
        return []
    
    # --- RIMOZIONE DI BLOCCHI PROMOZIONALI E IMMAGINI ---

    # # Rimuove moduli SPPageBuilder "latest news", caroselli o banner con soli link
    # for div in main_el.find_all("div", class_=lambda c: c and ("sppb-addon" in c or "sp-module" in c)):
    #     text = div.get_text(strip=True)
    #     links = div.find_all("a")
    #     # Se il blocco contiene molti link ma poco testo descrittivo, è promozionale
    #     if len(links) >= 3 and len(text) < 300:
    #         div.decompose()

    # # Rimuove anche <ul> o <section> che contengono solo link promozionali
    # for ul in main_el.find_all(["ul", "section"]):
    #     links = ul.find_all("a")
    #     text = ul.get_text(strip=True)
    #     if len(links) >= 3 and len(text) < 300:
    #         ul.decompose()

    # # Rimuove immagini e relativi contenitori senza testo utile
    # for tag in main_el.find_all(["img", "figure", "figcaption"]):
    #     tag.decompose()
    # for div in main_el.find_all("div"):
    #     if div.find("img") and not div.get_text(strip=True):
    #         div.decompose()


    # Rimuove i moduli laterali/promozionali (es. blocchi "Notizie dal Dipartimento")
    for mod in main_el.select("div.module-container.col-xs-12"):
        mod.decompose()

    main_html = str(main_el)
    
    #tmp_path = file_path.with_suffix(".main.html")
    #tmp_path.write_text(main_html, encoding="utf-8")

    #elements = partition_html(filename=str(tmp_path), include_page_breaks=False, languages=["ita", "eng"])

    elements = partition_html(
        text=main_html,                 # <-- usa text, NON filename (per evitare di riprocessare il file originale)
        include_page_breaks=False,
        languages=["ita", "eng"]
    )


    if not elements:
        text_fallback = main_el.get_text(separator="\n", strip=True)
        return [Document(
            page_content=text_fallback,
            metadata={
                "source_url": source_url,
                "doc_type": "html",
                "page_title": page_title,
                "element_type": "FallbackText",
                "lang": "ita",
                "crawl_ts": datetime.now(timezone.utc).isoformat(),
                "doc_id": sha(source_url),
            },
        )] if text_fallback else []

    return semantic_chunk(elements, source_url, page_title, doc_type="html", file_name=file_path.name)


# === PDF ===
def to_documents_from_pdf(file_path: Path, source_url: str) -> list[Document]:
    try:
        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            include_page_breaks=True,
            infer_table_structure=True,
            languages=["ita", "eng"]
        )
    except Exception as e:
        print(f"[WARN] partition_pdf fallito su {file_path}: {e}")
        return []
    
    if not elements:
        print(f"[WARN] Nessun elemento estratto da {file_path}")
        return []

    return semantic_chunk(elements, source_url, page_title=file_path.stem, doc_type="pdf", file_name=file_path.name)
