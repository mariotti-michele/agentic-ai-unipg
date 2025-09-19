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

for p in (RAW_HTML, RAW_PDF):
    p.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION = os.getenv("COLLECTION", "ing_info_mag_docs")

def slugify(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip())
    return s[:120] if s else "file"

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

async def check_if_pdf_url(url: str, client: httpx.AsyncClient) -> bool:
    """
    Controlla se un URL serve un PDF facendo una richiesta HEAD
    e verificando il Content-Type
    """
    try:
        # Prima controlla se l'URL finisce con .pdf (caso ovvio)
        if url.lower().endswith('.pdf'):
            return True
            
        # Altrimenti fai una richiesta HEAD per controllare il Content-Type
        response = await client.head(url, timeout=10.0)
        content_type = response.headers.get('content-type', '').lower()
        
        # Controlla se il Content-Type indica un PDF
        if 'application/pdf' in content_type:
            return True
            
        # Alcuni server potrebbero non supportare HEAD, prova GET con range limitato
        if response.status_code == 405:  # Method Not Allowed
            response = await client.get(url, headers={'Range': 'bytes=0-1023'}, timeout=10.0)
            content_type = response.headers.get('content-type', '').lower()
            if 'application/pdf' in content_type:
                return True
                
    except Exception as e:
        print(f"[WARN] Errore nel controllo PDF per {url}: {e}")
        return False
        
    return False

async def categorize_links(links: list[str]) -> tuple[list[str], list[str]]:
    """
    Categorizza i link in PDF e HTML controllando sia l'estensione che il Content-Type
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    
    pdf_links = []
    html_links = []
    
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
        headers=headers
    ) as client:
        
        # Processa i link in batch per efficienza
        semaphore = asyncio.Semaphore(10)  # Limita le richieste concorrenti
        
        async def check_link(link):
            async with semaphore:
                if await check_if_pdf_url(link, client):
                    pdf_links.append(link)
                elif link.startswith("http") and not link.lower().endswith(('.jpg', '.png', '.gif', '.jpeg')):
                    html_links.append(link)
        
        tasks = [check_link(link) for link in links]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    return pdf_links, html_links

async def scrape_page(url: str, same_domain_only: bool = False):
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
            if l:  # controlla subito che non sia None o stringa vuota
                try:
                    u = urlparse(l)
                    # se non c'è schema (http/https), completa l'URL
                    if not u.scheme:
                        l = urljoin(url, l)
                        u = urlparse(l)
                    
                    # aggiungi solo se:
                    # - o non filtriamo per dominio
                    # - oppure il dominio coincide con quello di origine
                    if (not same_domain_only) or (u.netloc and u.netloc == origin):
                        if l not in seen:
                            seen.add(l)
                            abs_links.append(l)
                except Exception as e:
                    print(f"[WARN] Link scartato {l}: {e}")
        
        # Categorizza i link controllando anche il Content-Type
        print(f"[INFO] Controllo Content-Type per {len(abs_links)} link...")
        pdf_links, html_links = await categorize_links(abs_links)
        
        return title, html_path, pdf_links, html_links

async def download_pdfs(urls: list[str]) -> list[Path]:
    saved = []
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf",
        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.unipg.it/",
    }
    
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=90.0,
        headers=headers
    ) as client:
        for u in urls:
            try:
                # Migliore generazione del nome file per URL dinamici
                parsed_url = urlparse(u)
                if parsed_url.path and not parsed_url.path.endswith('/'):
                    name_guess = slugify(Path(parsed_url.path).name or "document")
                else:
                    # Per URL dinamici, usa parametri della query o hash dell'URL
                    if parsed_url.query:
                        name_guess = slugify(parsed_url.query[:50]) or sha(u)[:12]
                    else:
                        name_guess = sha(u)[:12]
                
                if not name_guess.lower().endswith(".pdf"):
                    name_guess += ".pdf"
                
                out = RAW_PDF / name_guess
                
                if out.exists():
                    saved.append(out)
                    continue
                
                r = await client.get(u)
                r.raise_for_status()
                
                # Verifica che il contenuto sia effettivamente un PDF
                content_type = r.headers.get('content-type', '').lower()
                if 'application/pdf' not in content_type and not r.content.startswith(b'%PDF'):
                    print(f"[WARN] Il contenuto di {u} non sembra essere un PDF")
                    continue
                
                out.write_bytes(r.content)
                saved.append(out)
                print(f"[OK] Scaricato PDF: {u} -> {out.name}")
                
            except Exception as e:
                print(f"[WARN] PDF skip {u}: {e}")
    
    return saved

def to_documents_from_html(file_path: Path, source_url: str, page_title: str) -> list[Document]:
    elements = partition_html(filename=str(file_path), include_page_breaks=False, languages=["ita", "eng"])
    docs = []
    crawl_ts = datetime.now(timezone.utc).isoformat()
    
    for el in elements:
        text = str(el).strip()
        if text:
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
    elements = partition_pdf(filename=str(file_path), strategy="fast", include_page_breaks=True, languages=["ita", "eng"])
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
    client = QdrantClient(url=QDRANT_URL)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

async def main(seed_url: str, follow_internal_html: bool = False):
    print(f"[INFO] Crawling: {seed_url}")
    title, html_path, pdf_links, html_links = await scrape_page(seed_url)
    print(f"[INFO] Found PDFs: {len(pdf_links)} | HTML links: {len(html_links)}")
    
    if pdf_links:
        print("[INFO] PDF links found:")
        for pdf_link in pdf_links:
            print(f"  - {pdf_link}")
    
    pdf_files = await download_pdfs(pdf_links)
    
    # Normalizzazione in Document
    html_docs = to_documents_from_html(html_path, source_url=seed_url, page_title=title)
    pdf_docs = []
    for f in pdf_files:
        pdf_docs.extend(to_documents_from_pdf(f, source_url=seed_url))
    
    all_docs = html_docs + pdf_docs
    
    # Chunking
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