from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION", "ing_info_mag_docs")

client = QdrantClient(url=QDRANT_URL)

def dump_all_chunks():
    offset = None
    count = 0

    while True:
        points, next_page = client.scroll(
            collection_name=COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=50,        # scarica 50 alla volta (puoi aumentare)
            offset=offset
        )

        for p in points:
            count += 1
            print("="*80)
            print(f"ID: {p.id}")
            print("Chunk:", p.payload.get("page_content", ""))
            print("Metadati:", {k: v for k, v in p.payload.items() if k != "page_content"})

        if not next_page:
            break
        offset = next_page

    print(f"\n[INFO] Totale chunk trovati: {count}")

if __name__ == "__main__":
    dump_all_chunks()


# python dump_qdrant.py
#Questo ti stamperà tutti i chunk presenti nella collezione, uno dopo l’altro, con ID e metadati.