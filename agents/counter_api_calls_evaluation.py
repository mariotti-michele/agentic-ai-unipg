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
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI


# Contatore globale per le chiamate API
api_call_counter = {
    "total_calls": 0,
    "calls_by_key": {},
    "calls_by_metric": {},
    "total_tokens_input": 0,
    "total_tokens_output": 0,
    "total_cost_usd": 0.0
}


# Estimatore di token e costi per Gemini
def estimate_gemini_tokens(text: str) -> int:
    """Stima del numero di token (1 token ≈ 4 caratteri)"""
    return int(len(text) / 4)

def estimate_gemini_cost(input_tokens: int, output_tokens: int) -> float:
    """Costo stimato in USD per Gemini 2.5 Flash (ottobre 2025)"""
    input_cost = (input_tokens / 1_000_000) * 0.075  # $0.075 per 1M input token
    output_cost = (output_tokens / 1_000_000) * 0.30  # $0.30 per 1M output token
    return input_cost + output_cost

# Wrapper semplice che traccia le chiamate
class TrackedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """Wrapper per ChatGoogleGenerativeAI che traccia le chiamate API"""
    
    def _generate(self, *args, **kwargs):
        # Incrementa i contatori
        current_key = os.environ.get("GOOGLE_API_KEY", "unknown")[:6]
        api_call_counter["total_calls"] += 1
        api_call_counter["calls_by_key"][current_key] = \
            api_call_counter["calls_by_key"].get(current_key, 0) + 1
        
        print(f"[API CALL #{api_call_counter['total_calls']}] Key: {current_key}...")
        
        return super()._generate(*args, **kwargs)
    
    async def _agenerate(self, *args, **kwargs):
        # Incrementa i contatori (versione async)
        current_key = os.environ.get("GOOGLE_API_KEY", "unknown")[:6]
        api_call_counter["total_calls"] += 1
        api_call_counter["calls_by_key"][current_key] = \
            api_call_counter["calls_by_key"].get(current_key, 0) + 1
        
        print(f"[API CALL #{api_call_counter['total_calls']}] Key: {current_key}... (async)")
        
        return await super()._agenerate(*args, **kwargs)


def get_next_google_key():
    # crea un generatore ciclico di chiavi dalle GOOGLE_API_KEY
    keys = os.getenv("GOOGLE_API_KEY", "").split(",")
    keys = [k.strip() for k in keys if k.strip()]
    if not keys:
        raise ValueError("Nessuna GOOGLE_API_KEY valida trovata nel .env (usa GOOGLE_API_KEY=key1,key2,...)")
    for key in itertools.cycle(keys):
        yield key


google_key_gen = get_next_google_key()


def get_active_google_key():
    # restituisce la prossima chiave disponibile dal generatore
    return next(google_key_gen)


def get_llm():
    """Crea un LLM tracciato"""
    key = get_active_google_key()
    os.environ["GOOGLE_API_KEY"] = key
    os.environ["GOOGLE_API_USE_RETRY"] = "false"
    print(f"[INFO] Utilizzo API Key: {key[:6]}...")

    try:
        return TrackedChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0
        )
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e).lower() or "resourceexhausted" in str(e).lower():
            print(f"[WARN] Quota esaurita per {key[:6]}... passo alla prossima chiave.")
            return get_llm()
        else:
            print(f"[AVVISO] Errore con chiave {key[:6]} — passo alla successiva. Dettagli: {e}")
            return get_llm()


def print_api_stats():
    """Stampa statistiche delle chiamate API"""
    print("\n" + "="*60)
    print("STATISTICHE CHIAMATE API GEMINI")
    print("="*60)
    print(f"Totale chiamate: {api_call_counter['total_calls']}")
    print("\nChiamate per chiave API:")
    for key, count in api_call_counter['calls_by_key'].items():
        print(f"  - {key}...: {count} chiamate")
    print("="*60 + "\n")
    print(f"Token input totali stimati: {api_call_counter['total_tokens_input']}")
    print(f"Token output totali stimati: {api_call_counter['total_tokens_output']}")
    print(f"Costo totale stimato: ${api_call_counter['total_cost_usd']:.4f}")



def run_evaluation(version: str = "v1"):
    # Reset contatore all'inizio
    api_call_counter["total_calls"] = 0
    api_call_counter["calls_by_key"] = {}
    
    print(f"Avvio validazione RAG - versione {version}")
    base_dir = Path("evaluations") / version
    base_dir.mkdir(parents=True, exist_ok=True)

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

            # Ottieni risposta dal modello
            response = answer_query(q)
            if "Risposta:" in response:
                answer = response.split("Risposta:")[1].split("\n")[0].strip()
            else:
                answer = response.strip()

            # === Calcolo token stimati e costi ===
            prompt_text = q + "\n" + "\n".join(retrieved_ctx)
            input_tokens = estimate_gemini_tokens(prompt_text)
            output_tokens = estimate_gemini_tokens(answer)
            total_tokens = input_tokens + output_tokens
            total_cost = estimate_gemini_cost(input_tokens, output_tokens)

            # Aggiorna contatori globali
            api_call_counter["total_tokens_input"] += input_tokens
            api_call_counter["total_tokens_output"] += output_tokens
            api_call_counter["total_cost_usd"] += total_cost

            print(f"   → Token stimati: in={input_tokens}, out={output_tokens}, costo=${total_cost:.5f}")


            answers.append(answer)
            retrieved_contexts.append(retrieved_ctx)
        except Exception as e:
            print(f"Errore durante la domanda '{q}': {e}")
            answers.append("")
            retrieved_contexts.append([])

    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": retrieved_contexts,
        "answer": answers,
        "ground_truth": ground_truths
    })

    print("\nValutazione con Ragas...")
    # Crea un LLM tracciato
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

    # Stampa statistiche API
    print_api_stats()

    csv_path = base_dir / "ragas_results.csv"
    save_results_to_csv(csv_path, dataset, result_df, results)

    # Salva anche le statistiche API
    stats_path = base_dir / "api_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_calls": api_call_counter["total_calls"],
            "calls_by_key": api_call_counter["calls_by_key"]
        }, f, indent=2)
    print(f"Statistiche API salvate in: {stats_path}")

    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    if langsmith_key:
        print("\nInviando risultati a LangSmith...")
        client = Client()
        client.create_run(
            name=f"RAG Evaluation - UNIPG ({version})",
            run_type="chain",
            inputs={"questions": questions},
            outputs={
                "results": dict(results),
                "api_calls": api_call_counter["total_calls"]
            },
            metadata={
                "component": "ragas_evaluation",
                "api_stats": api_call_counter
            }
        )
        print("Risultati inviati a LangSmith.")
    else:
        print("Nessuna API key LangSmith trovata — risultati solo in locale.")


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

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
        # Aggiungi riga con statistiche API
        writer.writerow([
            timestamp,
            f"=== API CALLS: {api_call_counter['total_calls']} ===",
            "",
            "", "", "", "", ""
        ])

    print(f"Risultati individuali e medi salvati in: {csv_path}")


if __name__ == "__main__":
    version = os.getenv("RAG_EVAL_VERSION", "v1")
    run_evaluation(version)