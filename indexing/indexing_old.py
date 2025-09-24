import os, re, json, asyncio, hashlib
import uuid
import pdfplumber
import pandas as pd
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

        # raccogli link validi SOLO dentro <main>
        valid_links_debug = await page.eval_on_selector_all(
            "main a[href]",
            """
            els => els.map(e => ({href: e.href, html: e.outerHTML.substring(0,200)}))
            """
        )
        valid_links = [x["href"] for x in valid_links_debug]

        print(f"[DEBUG] Valid links ({len(valid_links)}):")
        for l in valid_links_debug:
            print(f"   [VALID] {l['href']} -- from: {l['html']}")

        
        await browser.close()
        
        def process_links(links):
            """Helper per processare e pulire una lista di link"""
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
            
            return abs_links
        
        # processa i link validi
        valid_abs_links = process_links(valid_links)
                
        # Categorizza solo i link validi
        print(f"[INFO] Controllo Content-Type per {len(valid_abs_links)} link validi...")
        pdf_links, html_links = await categorize_links(valid_abs_links)
        
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
                if parsed_url.query:
                    name_guess = slugify(parsed_url.path + "_" + parsed_url.query[:50]) or sha(u)[:12]
                else:
                    name_guess = slugify(parsed_url.path) or sha(u)[:12]

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

from bs4 import BeautifulSoup

def to_documents_from_html(file_path: Path, source_url: str, page_title: str) -> list[Document]:
    raw_html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")

    # prendi solo il contenuto dentro <main>
    main_el = soup.find("main")
    if not main_el:
        print(f"[WARN] Nessun <main> trovato in {source_url}, salto")
        return []

    # ricrea un file temporaneo solo con il contenuto del main
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

def clean_pdf_table(table):
    """
    Pulisce e normalizza una tabella grezza estratta da pdfplumber
    """
    # Rimuove righe completamente vuote
    table = [row for row in table if any(cell for cell in row)]

    if not table:
        return pd.DataFrame()

    # Usa la PRIMA riga non vuota come header
    headers = [str(h).strip() if h else "" for h in table[0]]

    # Se ci sono header multi-riga, uniscili con quelli successivi
    if "" in headers or "null" in headers:
        # prova a prendere le prime 2 righe come intestazioni e unirle
        headers = [
            ((str(table[0][i]) or "") + " " + (str(table[1][i]) or "")).strip()
            for i in range(len(table[0]))
        ]
        data = table[2:]
    else:
        data = table[1:]

    # Crea DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Rimuove colonne vuote
    df = df.loc[:, df.columns.notna()]
    df = df.dropna(how="all", axis=1)

    # Normalizza i nomi delle colonne
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]

    return df

def extract_tables_pdfplumber(file_path: Path, source_url: str) -> list[dict]:
    """
    Estrae tabelle da un PDF e le restituisce come lista di record JSON
    (ogni riga = dict con intestazioni).
    """
    records = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                df = clean_pdf_table(table)
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    records.append({
                        "row": row.to_dict(),
                        "page_number": page_num,
                        "file_name": file_path.name,
                        "table_index": t_idx
                    })
    return records



def to_documents_from_pdf(file_path: Path, source_url: str) -> list[Document]:
    # Estraggo testo con unstructured
    elements = partition_pdf(filename=str(file_path), strategy="hi_res",
                             include_page_breaks=True, infer_table_structure=True,
                             languages=["ita", "eng"])
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

    # Estraggo tabelle strutturate con pdfplumber
    tables = extract_tables_pdfplumber(file_path, source_url)
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


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    # aggiungi ID deterministico
    for i, c in enumerate(chunks):
        base_id = sha(c.metadata["source_url"])
        content_id = sha(c.page_content)
        c.metadata["chunk_id"] = f"{base_id}_{i}_{content_id[:8]}"
    return chunks


def build_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    client = QdrantClient(url=QDRANT_URL)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )


async def crawl(seed_url: str, max_depth: int = 2):
    visited = set()
    to_visit = [(seed_url, 0)]
    all_docs = []

    while to_visit:
        url, depth = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"\n[INFO] Crawling ({depth}/{max_depth}): {url}")
        try:
            title, html_path, pdf_links, html_links = await scrape_page(url)
            print(f"[DEBUG] Link trovati nel <main> della pagina: {url}")
            print(f"  - PDF: {len(pdf_links)}")
            for l in pdf_links:
                print(f"    [PDF] {l}")
            print(f"  - HTML: {len(html_links)}")
            for l in html_links:
                print(f"    [HTML] {l}")

        except Exception as e:
            print(f"[WARN] Skip {url}: {e}")
            continue

        pdf_files = await download_pdfs(pdf_links)
        html_docs = to_documents_from_html(html_path, source_url=url, page_title=title)
        pdf_docs = []
        for f in pdf_files:
            pdf_docs.extend(to_documents_from_pdf(f, source_url=url))
        all_docs.extend(html_docs + pdf_docs)

        if depth < max_depth:
            for link in html_links:
                if link not in visited:
                    print(f"[DEBUG] -> Da visitare (depth {depth+1}): {link}")
                    to_visit.append((link, depth + 1))

    return all_docs


async def main(seed_url: str, follow_internal_html: bool = False, max_depth: int = 2):
    all_docs = await crawl(seed_url, max_depth)

    # Chunking
    chunks = chunk_documents(all_docs)
    print(f"[INFO] Chunks: {len(chunks)}")
    
    # Upsert su Qdrant
    vs = build_vectorstore()
    #ids = [c.metadata["chunk_id"] for c in chunks]
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    vs.add_documents(chunks, ids=ids)   # overwrite se già presenti
    print(f"[OK] Upsert completato")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="URL della pagina accademica di partenza")
    ap.add_argument("--depth", type=int, default=2, help="Profondità massima del crawling")
    args = ap.parse_args()
    asyncio.run(main(args.url, max_depth=args.depth))