import tiktoken

# === Prompt template ===
prompt_template = """Sei un assistente accademico.
Hai accesso a estratti di documenti ufficiali.

Usa SOLO il contesto fornito per rispondere.
Se il contesto contiene il termine o l'argomento richiesto, indica che è presente e copia il testo più rilevante.
Non aggiungere nulla di tuo e non inventare.

Se davvero non ci sono riferimenti nemmeno parziali, rispondi esattamente: "Non presente nei documenti".

Domanda: {question}

Contesto:
{context}

Risposta:"""

# === Domanda ===
question = "Qual è l'orario delle lezioni di Intelligent and Secure Networks e quali sono le date degli appelli d'esame?"

# === Contesto: 5 chunk identici ===
chunk_text = """I metodi di valutazione di questo insegnamento cercano di quantificare le conoscenze teoriche acquisite dallo studente, nonché la sue capacità di applicare tali conoscenze per la risoluzione di problemi applicativi. I tipi di prove previste per la valutazione sono descritti qui di seguito. - Prova scritta di natura teorica Durata: 60 minuti Punteggio: 10/30 Obiettivo: accertare le conoscenze sui concetti teorici impartiti nell'insegnamento. - Prova al calcolatore Durata: 120 minuti Punteggio: 20/30 Obiettivo: accertare le abilità pratiche acquisite in relazione alle tematiche del corso Programma esteso PARTE I - Introduzione all'ingegneria del software - Qualità e principi del software - Modelli di produzione del software - Ingegneria dei requisiti - Progettazione architetturale - Programmazione a oggetti e design patterns - Test del software PARTE II - Introduzione a sistemi software basati su AI - Ingegneria dei requisiti per sistemi basati su AI - Progettazione, deployment e automazione di componenti software basati su AI - Responsible Engineering Obiettivi Agenda 2030 per lo sviluppo sostenibile Industria, innovazione e infrastrutture; Ridurre le disuguaglianze"""
context = "\n\n".join([chunk_text for _ in range(5)])

# === Funzione per contare token ===
def count_tokens_openai(text: str, model: str = "gpt-4o-mini"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # fallback generico
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# === Costruzione prompt ===
prompt = prompt_template.format(context=context, question=question)

# === Conteggio token ===
input_tokens = count_tokens_openai(prompt, model="gpt-4o-mini")
print("📊 Token input stimati:", input_tokens)

# === Stima output (ipotizziamo 100 token medi) ===
output_tokens = 100
print("📊 Token output stimati (media):", output_tokens)
print("📊 Totale stimato per query:", input_tokens + output_tokens)

# === Stampa preview prompt per verifica ===
print("\n--- Preview del prompt ---\n")
print(prompt[:800] + "...\n[TRONCATO]")
