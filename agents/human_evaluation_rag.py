import json
import os
from pathlib import Path
#from baseline_rag_agent import answer_query, embeddings, vectorstore
from filter_rag import embeddings, vectorstore, answer_query_dense, answer_query_tfidf, hybrid_search
from filter_rag import embeddings, vectorstore, answer_query_dense, answer_query_tfidf, answer_query_bm25, hybrid_search


def run_manual_eval():
    VALIDATION_DIR = Path(__file__).resolve().parent / "validation_set"
    print(f"Caricamento dataset da: {VALIDATION_DIR}")

    validation_data = []
    for json_file in sorted(VALIDATION_DIR.glob("*.json")):
        print(f"  → Trovato file: {json_file.name}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    validation_data.extend(content)
                else:
                    print(f"Il file {json_file.name} non contiene una lista JSON valida, ignorato.")
        except Exception as e:
            print(f"Errore nel file {json_file.name}: {e}")

    print(f"Totale domande caricate: {len(validation_data)}\n")

    for i, d in enumerate(validation_data, start=1):
        q = d["question"]
        print("="*80)
        print(f"[{i}] Domanda: {q}\n")

        try:
            vec = embeddings.embed_query(q)
            docs = vectorstore.similarity_search_by_vector(vec, k=5)
            retrieved_ctx = [d.page_content for d in docs]

            #response = answer_query(q)
            #print("→ Retrieval Dense")
            response_dense = answer_query_dense(q)

            #print("→ Retrieval Sparse (TF-IDF)")
            response_sparse = answer_query_tfidf(q)

            #print("→ Retrieval BM25")
            response_bm25 = answer_query_bm25(q)

            #print("→ Retrieval Ibrido (Hybrid)")
            response_hybrid = hybrid_search(q, alpha=0.6)

            print(response_dense, "\n")
            print(response_sparse, "\n")
            print(response_bm25, "\n")
            print(response_hybrid, "\n")



            # # pulizia risposta
            # if "Risposta:" in response:
            #     answer = response.split("Risposta:")[1].split("\n")[0].strip()
            # else:
            #     answer = response.strip()

            # print(f"Risposta generata:\n{answer}\n")


            print("Contesti recuperati:")
            for c in retrieved_ctx:
                print("-", c[:200], "...")
        except Exception as e:
            print(f"Errore durante la domanda '{q}': {e}")
            print("Risposta generata: [ERRORE]\n")

    print("="*80)
    print("Completato. Tutte le risposte sono state stampate.")


if __name__ == "__main__":
    run_manual_eval()
