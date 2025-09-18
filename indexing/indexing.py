import os, re, json, asyncio, hashlib
from pathlib import Path
from urllib.parse import urlparse, urljoin
from datetime import datetime, timezone

from dotenv import load_dotenv
from playwright.async_api import async_playwright
import httpx

from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_HTML = DATA_DIR / "raw" / "html"
RAW_PDF = DATA_DIR / "raw" / "pdf"
for p in (RAW_HTML, RAW_PDF): p.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION = os.getenv("COLLECTION", "academic_docs")

def slugify(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip())
    return s[:120] if s else "file"

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

async def scrape_page(url: str, same_domain_only: bool = True, max_links: int = 60):
    """
    Rende la pagina, salva l'HTML e raccoglie i link (in particolare PDF).
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=90000)
        title = await page.title()
        html = await page.content()

        # salva HTML raw
        fname = slugify(urlparse(url).path or "index") + ".html"
        html_path = RAW_HTML / fname
        html_path.write_text(html, encoding="utf-8")

        # raccogli link assoluti
        links = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => e.href)"
        )
        await browser.close()

    # pulizia & filtro
    origin = urlparse(url).netloc
    seen = set()
    abs_links = []
    for l in links:
        if not l: continue
        try:
            u = urlparse(l)
            if not u.scheme:
                l = urljoin(url, l)
                u = urlparse(l)
            if same_domain_only and u.netloc and u.netloc != origin:
                continue
            if l not in seen:
                seen.add(l)
                abs_links.append(l)
        except Exception:
            pass

    # PDF + sottopagine HTML della stessa sezione (facoltativo)
    pdf_links = [l for l in abs_links if l.lower().endswith(".pdf")]
    html_links = [l for l in abs_links if (l.startswith("http") and not l.lower().endswith(".pdf"))]

    # limiti di cortesia
    pdf_links = pdf_links[:max_links]
    html_links = html_links[:max_links]

    return title, html_path, pdf_links, html_links

async def download_pdfs(urls: list[str]) -> list[Path]:
    saved = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=90.0) as client:
        for u in urls:
            try:
                name_guess = slugify(Path(urlparse(u).path).name or "document") or "document"
                if not name_guess.lower().endswith(".pdf"):
                    name_guess += ".pdf"
                out = RAW_PDF / name_guess
                if out.exists():
                    saved.append(out); continue
                r = await client.get(u)
                r.raise_for_status()
                out.write_bytes(r.content)
                saved.append(out)
            except Exception as e:
                print(f"[WARN] PDF skip {u}: {e}")
    return saved

def to_documents_from_html(file_path: Path, source_url: str, page_title: str) -> list[Document]:
    # Unstructured: scompone in elementi tipizzati (Title, NarrativeText, ListItem, Table, ...)
    elements = partition_html(filename=str(file_path), include_page_breaks=False)
    docs = []
    crawl_ts = datetime.now(timezone.utc).isoformat()
    for el in elements:
        text = str(el).strip()
        if not text: 
            continue
        meta = getattr(el, "metadata", None)
        meta = meta.to_dict() if meta is not None else {}
        doc = Document(
            page_content=text,
            metadata={
                "source_url": source_url,
                "doc_type": "html",
                "page_title": page_title,
                "section": meta.get("category", None),
                "lang": meta.get("languages", None),
                "crawl_ts": crawl_ts,
                "doc_id": sha(source_url),
            },
        )
        docs.append(doc)
    return docs

def to_documents_from_pdf(file_path: Path, source_url: str) -> list[Document]:
    # strategy="fast" usa pdfminer; se serve OCR puoi passare "hi_res"/"ocr_only"
    elements = partition_pdf(filename=str(file_path), strategy="fast", include_page_breaks=True)
    docs = []
    crawl_ts = datetime.now(timezone.utc).isoformat()
    for el in elements:
        text = str(el).strip()
        if not text:
            continue
        meta = getattr(el, "metadata", None)
        meta = meta.to_dict() if meta is not None else {}
        doc = Document(
            page_content=text,
            metadata={
                "source_url": source_url,
                "doc_type": "pdf",
                "page_number": meta.get("page_number", None),
                "file_name": file_path.name,
                "crawl_ts": crawl_ts,
                "doc_id": sha(source_url),
            },
        )
        docs.append(doc)
    return docs

def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )
    return splitter.split_documents(docs)

def build_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    client = QdrantClient(url=QDRANT_URL)  # se usi API key, aggiungi api_key=...
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

async def main(seed_url: str, follow_internal_html: bool = False):
    print(f"[INFO] Crawling: {seed_url}")
    title, html_path, pdf_links, html_links = await scrape_page(seed_url)
    print(f"[INFO] Found PDFs: {len(pdf_links)} | HTML links: {len(html_links)}")

    pdf_files = await download_pdfs(pdf_links)

    # Normalizzazione in Document
    html_docs = to_documents_from_html(html_path, source_url=seed_url, page_title=title)
    pdf_docs = []
    for f in pdf_files:
        # salviamo nei metadata il link da cui è venuto, se identico al file name lo deduciamo
        pdf_docs.extend(to_documents_from_pdf(f, source_url=seed_url))

    all_docs = html_docs + pdf_docs

    # Chunking (v0.2 splitters)
    chunks = chunk_documents(all_docs)
    print(f"[INFO] Chunks: {len(chunks)}")

    # Upsert su Qdrant
    vs = build_vectorstore()
    ids = vs.add_documents(chunks)
    print(f"[OK] Upsert completato. Esempio ID: {ids[:3]}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="URL della pagina accademica di partenza")
    args = ap.parse_args()
    asyncio.run(main(args.url))
