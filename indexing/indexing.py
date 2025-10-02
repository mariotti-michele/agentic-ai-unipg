import os, asyncio, uuid
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from crawling import crawl
from scraping import sha

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION = os.getenv("COLLECTION", "ing_info_mag_docs")


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    clean_chunks = []
    for i, c in enumerate(chunks):
        text = c.page_content.strip()
        #if not text or len(text) < 20:
        #    print(f"[SKIP] Chunk scartato (vuoto o troppo breve): {repr(text[:50])}")
        #    continue

        base_id = sha(c.metadata["source_url"])
        content_id = sha(c.page_content)
        c.metadata["chunk_id"] = f"{base_id}_{i}_{content_id[:8]}"
        clean_chunks.append(c)
        
    return clean_chunks

def build_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    client = QdrantClient(url=QDRANT_URL)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )


async def main(seed_url: str, max_depth: int = 0, is_download_pdf_active: bool = True):
    all_docs = await crawl(seed_url, max_depth, is_download_pdf_active)

    chunks = chunk_documents(all_docs)
    print(f"[INFO] Chunks finali da inserire: {len(chunks)}")

    if not chunks:
        print("[WARN] Nessun chunk valido trovato, niente da upsertare.")
        return

    vs = build_vectorstore()
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    vs.add_documents(chunks, ids=ids)
    print(f"[OK] Upsert completato: {len(chunks)} punti inseriti")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="URL della pagina di partenza")
    ap.add_argument("--depth", type=int, default=0, help="Profondità massima del crawling")
    ap.add_argument(
        "--pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scarica i PDF (usa --no-pdf per disabilitare)",
    )
    args = ap.parse_args()
    asyncio.run(main(args.url, max_depth=args.depth, is_download_pdf_active=args.pdf))