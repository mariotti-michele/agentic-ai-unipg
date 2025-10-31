#  HYBRID DOC STRUCTURE PLUS FIXED SIZE CHUNKING = FIXED SIZE CHUNKING INDEXING SCRIPT

import os, asyncio, uuid
from dotenv import load_dotenv
from pathlib import Path

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

LINKS_FILE = Path(__file__).resolve().parent / "links.txt"


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )
    chunks = []
    for d in docs:
        if d.metadata.get("element_type") in ["Table", "pdf-table", "ScheduleJSON"]:
            chunks.append(d)
        else:
            chunks.extend(splitter.split_documents([d]))

    clean_chunks = []
    for i, c in enumerate(chunks):
        text = c.page_content.strip()
        base_id = sha(c.metadata["source_url"])
        content_id = sha(c.page_content)
        c.metadata["chunk_id"] = f"{base_id}_{i}_{content_id[:8]}"
        clean_chunks.append(c)

    return clean_chunks


def build_vectorstore(collection_name: str):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    client = QdrantClient(url=QDRANT_URL)

    # --- Creazione automatica collezione se non esiste ---
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        print(f"[INFO] Creazione nuova collezione Qdrant: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": 768, "distance": "Cosine"},
        )
    else:
        print(f"[INFO] Collezione già esistente: {collection_name}")

    # --- Costruzione vectorstore LangChain ---
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )



# === Parsing del file links.txt in blocchi ===
def parse_links_file() -> dict[str, list[tuple[str, dict]]]:
    """
    Ritorna un dizionario: { 'Nome blocco' : [(url, opzioni), ...], ... }
    """
    blocks = {}
    current_block = None

    for line in LINKS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            current_block = line.lstrip("#").strip()
            # Sostituisci spazi e caratteri speciali con underscore (compatibile con Qdrant)
            sanitized_name = (
                current_block.replace(" ", "_")
                .replace("/", "_")
                .replace("-", "_")
                .replace(".", "_")
            )
            blocks[sanitized_name] = []
        else:
            if current_block is None:
                continue

            parts = line.split()
            url = parts[0]
            options = {"depth": 0, "pdf": True}

            if "--no-pdf" in parts:
                options["pdf"] = False
            for i, p in enumerate(parts):
                if p == "--depth" and i + 1 < len(parts):
                    options["depth"] = int(parts[i + 1])

            blocks[sanitized_name].append((url, options))

    return blocks


# === Main per ogni blocco ===
async def process_block(collection_name: str, urls_with_opts: list[tuple[str, dict]]):
    print(f"\n\n=== [BLOCCO: {collection_name}] ===")
    print(f"[INFO] Creazione/aggiornamento collezione: {collection_name}")

    all_docs = []
    for url, opts in urls_with_opts:
        print(f"\n>>> Processando: {url} (depth={opts['depth']}, pdf={opts['pdf']})")
        docs = await crawl(url, opts["depth"], opts["pdf"])
        all_docs.extend(docs)

    chunks = chunk_documents(all_docs)
    print(f"[INFO] Chunks finali da inserire: {len(chunks)}")

    if not chunks:
        print(f"[WARN] Nessun chunk valido per il blocco {collection_name}")
        return

    vs = build_vectorstore(collection_name)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    vs.add_documents(chunks, ids=ids)
    print(f"[OK] Upsert completato: {len(chunks)} punti inseriti nella collezione '{collection_name}'")


async def main():
    blocks = parse_links_file()
    if not blocks:
        print("[ERRORE] Nessun blocco trovato in links.txt")
        return

    print(f"[INFO] Trovati {len(blocks)} blocchi nel file links.txt")

    for collection_name, urls_with_opts in blocks.items():
        await process_block(collection_name, urls_with_opts)


if __name__ == "__main__":
    asyncio.run(main())
