import json
import os
import csv
from datetime import datetime
from pathlib import Path
from datasets import Dataset
from ragas.metrics import faithfulness, context_precision
from ragas.metrics._answer_relevance import answer_relevancy
from ragas import evaluate
from langsmith import Client
from retrieval_agent import answer_query, embeddings, vectorstore
from pathlib import Path


def run_evaluation(version: str = "v1"):
    """
    Esegue la valutazione automatica del Retrieval Agent di UNIPG
    leggendo un dataset con colonne: question, reference_context, ground_truth, document.
    I risultati vengono salvati in evaluations/<version>/ragas_results.csv.
    """
    print(f"Avvio validazione RAG - versione {version}")
    base_dir = Path("evaluations") / version
    base_dir.mkdir(parents=True, exist_ok=True)

    VALIDATION_DIR = Path(__file__).resolve().parent / "validation_set"
    print(f"Caricamento dataset da: {VALIDATION_DIR}")

    data = []
    for json_file in sorted(VALIDATION_DIR.glob("*.json")):
        print(f"  → Trovato file: {json_file.name}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    data.extend(content)
                else:
                    print(f"Il file {json_file.name} non contiene una lista JSON valida, ignorato.")
        except Exception as e:
            print(f"Errore nel file {json_file.name}: {e}")

    print(f"Totale domande caricate: {len(data)}")

    questions = [d["question"] for d in data]
    reference_contexts = [d.get("reference_context", []) for d in data]
    ground_truths = [d.get("ground_truth", None) for d in data]
    documents = [d.get("document", None) for d in data]

    answers, retrieved_contexts = [], []

    print(f"Generazione risposte con il Retrieval Agent ({len(questions)} domande)...")
    for i, q in enumerate(questions, start=1):
        print(f" → [{i}/{len(questions)}] {q}")
        try:
            # Recupera contesti effettivi dal retriever
            vec = embeddings.embed_query(q)
            docs = vectorstore.similarity_search_by_vector(vec, k=5)
            retrieved_ctx = [d.page_content for d in docs]

            # Ottieni risposta dal modello
            response = answer_query(q)
            if "Risposta:" in response:
                answer = response.split("Risposta:")[1].split("\n")[0].strip()
            else:
                answer = response.strip()

            answers.append(answer)
            retrieved_contexts.append(retrieved_ctx)
        except Exception as e:
            print(f"Errore durante la domanda '{q}': {e}")
            answers.append("")
            retrieved_contexts.append([])

    # === Crea dataset per RAGAS ===
    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": retrieved_contexts,       # contesto trovato dal retriever
        "answer": answers,                    # risposta del modello
        "ground_truth": ground_truths,        # risposta corretta
        "reference_context": reference_contexts,  # contesto di riferimento
        "document": documents                 # testo documento (opzionale)
    })

    print("\nValutazione con Ragas...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print("\nRISULTATI RAGAS:")
    for k, v in results.items():
        print(f" - {k}: {v:.3f}")

    # === Salva CSV versionato ===
    csv_path = base_dir / "ragas_results.csv"
    save_results_to_csv(csv_path, questions, answers, results)

    # === Invia risultati a LangSmith (opzionale) ===
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    if langsmith_key:
        print("\nInviando risultati a LangSmith...")
        client = Client()
        client.create_run(
            name=f"RAG Evaluation - UNIPG ({version})",
            run_type="evaluation",
            inputs={"questions": questions},
            outputs={"results": dict(results)},
        )
        print("Risultati inviati a LangSmith.")
    else:
        print("ℹNessuna API key LangSmith trovata — risultati solo in locale.")


def save_results_to_csv(csv_path: Path, questions, answers, metrics):
    """
    Salva i risultati di valutazione in un file CSV versionato.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "question", "answer",
                "faithfulness", "answer_relevancy", "context_precision"
            ])
        for i, q in enumerate(questions):
            writer.writerow([
                timestamp,
                q,
                answers[i],
                f"{metrics['faithfulness']:.3f}",
                f"{metrics['answer_relevancy']:.3f}",
                f"{metrics['context_precision']:.3f}",
            ])

    print(f"Risultati salvati in: {csv_path}")


if __name__ == "__main__":
    version = os.getenv("RAG_EVAL_VERSION", "v1")
    run_evaluation(version)
