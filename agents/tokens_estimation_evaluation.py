import json
import os
import csv
import itertools
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.metrics._answer_relevance import answer_relevancy
from ragas import evaluate
from langsmith import Client
from baseline_rag_agent import answer_query, embeddings, vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI

import tiktoken


# === CONTATORE GLOBALE TOKEN (simulazione costi OpenAI) ===
token_counter = {
    "input_tokens": 0,
    "output_tokens": 0,
    "calls": 0
}


def count_tokens(text: str, model: str = "gpt-4o-mini"):
    """Conta quanti token OpenAI verrebbero usati per un testo"""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# === Gestione API Keys Gemini ===
def get_next_google_key():
    keys = os.getenv("GOOGLE_API_KEY", "").split(",")
    keys = [k.strip() for k in keys if k.strip()]
    if not keys:
        raise ValueError("Nessuna GOOGLE_API_KEY valida trovata nel .env (usa GOOGLE_API_KEY=key1,key2,...)")
    for key in itertools.cycle(keys):
        yield key


google_key_gen = get_next_google_key()


def get_active_google_key():
    return next(google_key_gen)


# === Inizializzazione LLM Gemini ===
def get_llm():
    key = get_active_google_key()
    os.environ["GOOGLE_API_KEY"] = key
    os.environ["GOOGLE_API_USE_RETRY"] = "false"
    print(f"[INFO] Utilizzo API Key: {key[:6]}...")

    try:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e).lower() or "resourceexhausted" in str(e).lower():
            print(f"[WARN] Quota esaurita per {key[:6]}... passo alla prossima chiave.")
            return get_llm()
        else:
            print(f"[AVVISO] Errore con chiave {key[:6]} — passo alla successiva. Dettagli: {e}")
            return get_llm()


# === Funzione principale ===
def run_evaluation(version: str = "v1"):
    print(f"Avvio validazione RAG - versione {version}")
    base_dir = Path("evaluations") / version
    base_dir.mkdir(parents=True, exist_ok=True)

    VALIDATION_DIR = Path(__file__).resolve().parent / "validation_set"
    print(f"Caricamento dataset da: {VALIDATION_DIR}")

    # Caricamento dataset JSON
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

    print(f"Totale domande caricate: {len(validation_data)}")

    questions = [d["question"] for d in validation_data]
    ground_truths = [d.get("ground_truth", None) for d in validation_data]

    answers, retrieved_contexts = [], []

    print(f"Generazione risposte con il Retrieval Agent ({len(questions)} domande)...")
    for i, q in enumerate(questions, start=1):
        print(f" → [{i}/{len(questions)}] {q}")
        try:
            # Recupera contesti effettivi dal retriever
            vec = embeddings.embed_query(q)
            docs = vectorstore.similarity_search_by_vector(vec, k=5)
            retrieved_ctx = [d.page_content for d in docs]

            # === Stima token input per questa domanda (prompt + contesto) ===
            prompt_preview = f"Domanda: {q}\nContesti: {retrieved_ctx}"
            tokens_input = count_tokens(prompt_preview)
            token_counter["input_tokens"] += tokens_input
            token_counter["calls"] += 1

            # Ottieni risposta dal modello
            response = answer_query(q)
            if "Risposta:" in response:
                answer = response.split("Risposta:")[1].split("\n")[0].strip()
            else:
                answer = response.strip()

            # === Stima token output per la risposta ===
            tokens_output = count_tokens(answer)
            token_counter["output_tokens"] += tokens_output

            answers.append(answer)
            retrieved_contexts.append(retrieved_ctx)

        except Exception as e:
            print(f"Errore durante la domanda '{q}': {e}")
            answers.append("")
            retrieved_contexts.append([])

    # Crea dataset per RAGAS
    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": retrieved_contexts,
        "answer": answers,
        "ground_truth": ground_truths
    })

    print("\nValutazione con Ragas...")
    llm = get_llm()

    metrics_list = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics_list,
        llm=llm,
        embeddings=embeddings,
    )

    result_df = result.to_pandas()
    results = result_df.mean(numeric_only=True).to_dict()

    print("\nRISULTATI RAGAS GLOBALI:")
    for k, v in results.items():
        print(f" - {k}: {v:.3f}")

    csv_path = base_dir / "ragas_results.csv"
    save_results_to_csv(csv_path, dataset, result_df, results)

    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    if langsmith_key:
        print("\nInviando risultati a LangSmith...")
        client = Client()
        client.create_run(
            name=f"RAG Evaluation - UNIPG ({version})",
            run_type="chain",
            inputs={"questions": questions},
            outputs={"results": dict(results)},
            metadata={"component": "ragas_evaluation"}
        )
        print("Risultati inviati a LangSmith.")
    else:
        print("Nessuna API key LangSmith trovata — risultati solo in locale.")

    # === RIEPILOGO TOKEN ===
    print("\n===== STATISTICHE TOKEN (STIMA OPENAI) =====")
    print(f"Totale chiamate LLM (simulate): {token_counter['calls']}")
    print(f"Totale token input stimati: {token_counter['input_tokens']:,}")
    print(f"Totale token output stimati: {token_counter['output_tokens']:,}")
    totale = token_counter['input_tokens'] + token_counter['output_tokens']
    print(f"Totale complessivo stimato: {totale:,} token")

    # (Opzionale) stima del costo GPT-4o-mini
    cost_input = token_counter['input_tokens'] / 1_000_000 * 0.15
    cost_output = token_counter['output_tokens'] / 1_000_000 * 0.60
    cost_total = cost_input + cost_output
    print(f"💰 Costo stimato con GPT-4o-mini: ${cost_total:.3f}")

    # Salva su CSV per analisi successive
    csv_token_path = Path('evaluations') / version / 'token_usage.csv'
    with open(csv_token_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'input_tokens', 'output_tokens', 'calls', 'total_tokens', 'cost_est_usd'])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            token_counter['input_tokens'],
            token_counter['output_tokens'],
            token_counter['calls'],
            totale,
            round(cost_total, 4)
        ])
    print(f"📊 Token usage salvato in: {csv_token_path}")


# === Salvataggio risultati RAGAS ===
def save_results_to_csv(csv_path: Path, dataset: Dataset, result_df, metrics):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "question", "answer", "faithfulness",
                "answer_relevancy", "context_precision", "context_recall",
                "answer_correctness"
            ])

        for i in range(len(dataset)):
            row = result_df.iloc[i]
            writer.writerow([
                timestamp,
                dataset[i]["question"],
                dataset[i]["answer"],
                f"{row['faithfulness']:.3f}",
                f"{row['answer_relevancy']:.3f}",
                f"{row['context_precision']:.3f}",
                f"{row['context_recall']:.3f}",
                f"{row['answer_correctness']:.3f}",
            ])

        writer.writerow([
            timestamp,
            "=== MEDIA TOTALE ===",
            "",
            f"{metrics['faithfulness']:.3f}",
            f"{metrics['answer_relevancy']:.3f}",
            f"{metrics['context_precision']:.3f}",
            f"{metrics['context_recall']:.3f}",
            f"{metrics['answer_correctness']:.3f}",
        ])

    print(f"Risultati individuali e medi salvati in: {csv_path}")


if __name__ == "__main__":
    version = os.getenv("RAG_EVAL_VERSION", "v1")
    run_evaluation(version)
