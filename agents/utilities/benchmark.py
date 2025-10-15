import time, requests

OLLAMA_URL = "http://ollama:11434"

MODELS = ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"]
PROMPTS = [
    "Spiega in 3 frasi cos'è un database vettoriale.",
    "Scrivi un riassunto di 5 righe sulla rivoluzione francese.",
    "Risolvi: se x+5=12, quanto vale x?"
]

def run_benchmark(model, prompt):
    start = time.time()
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    elapsed = time.time() - start
    data = r.json()
    output = data.get("response", "").strip()
    return elapsed, output

for model in MODELS:
    print(f"\n=== Benchmark {model} ===")
    for p in PROMPTS:
        t, out = run_benchmark(model, p)
        print(f"[{model}] Tempo: {t:.2f}s | Prompt: {p}\n Risposta: {out}\n")
