import os
import argparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION = os.getenv("COLLECTION", "ing_info_mag_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = "http://localhost:11434"

client = QdrantClient(url=QDRANT_URL)

# --- Ricerca per KEYWORD ---------------------------------------------------
def search_by_keyword(keyword: str, limit: int = 50):
    results = []
    offset = None

    while True:
        points, next_page = client.scroll(
            collection_name=COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=limit,
            offset=offset
        )

        for p in points:
            text = p.payload.get("page_content", "")
            if keyword.lower() in text.lower():
                results.append({
                    "id": p.id,
                    "chunk": text,
                    "metadata": {k: v for k, v in p.payload.items() if k != "page_content"}
                })

        if not next_page:
            break
        offset = next_page

    return results

# --- Ricerca SEMANTICA -----------------------------------------------------
def search_semantic(query: str, k: int = 5):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vs = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )
    results = vs.similarity_search(query, k=k)
    return results

# --- MAIN -----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query su Qdrant")
    parser.add_argument("--mode", choices=["keyword", "semantic"], required=True,
                        help="Tipo di ricerca: 'keyword' o 'semantic'")
    parser.add_argument("--query", required=True, help="La query da cercare")
    parser.add_argument("--limit", type=int, default=5,
                        help="Numero di risultati da mostrare (0 = tutti)")
    parser.add_argument("--show-full", action="store_true",
                        help="Mostra tutto il chunk invece che i primi 300 caratteri")
    args = parser.parse_args()

    if args.mode == "keyword":
        hits = search_by_keyword(args.query)

        # se limit = 0 → mostra tutti
        limit = len(hits) if args.limit == 0 else args.limit

        for h in hits[:limit]:
            print("="*80)
            print("ID:", h["id"])
            if args.show_full:
                print("Chunk:", h["chunk"])
            else:
                print("Chunk:", h["chunk"][:300], "...")
            print("Metadati:", h["metadata"])

    elif args.mode == "semantic":
        results = search_semantic(args.query, k=(args.limit if args.limit > 0 else 50))

        for r in results:
            print("="*80)
            if args.show_full:
                print("Chunk:", r.page_content)
            else:
                print("Chunk:", r.page_content[:300], "...")
            print("Metadati:", r.metadata)



# Ricerca per keyword:
# python query_qdrant.py --mode keyword --query "regolamento"
# Ricerca semantica:
# python query_qdrant.py --mode semantic --query "regolamento universitario" --limit 10

# Ricerca keyword con chunk interi:
# python query_qdrant.py --mode keyword --query "statuto" --limit 5 --show-full

# Ricerca semantica con chunk interi:
# python query_qdrant.py --mode semantic --query "regolamento universitario" --limit 3 --show-full


# python query_qdrant.py --mode keyword --query "statuto" --show-full --limit 0