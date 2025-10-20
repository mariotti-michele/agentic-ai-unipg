import os, json
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate

#llama locale:
from langchain_ollama import OllamaLLM

#gemini:
from langchain_google_genai import ChatGoogleGenerativeAI

#llama 3.3 70b api:
from langchain_google_vertexai import ChatVertexAI
from google.oauth2 import service_account

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# BM25
from rank_bm25 import BM25Okapi



if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
    creds_dict = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    creds = service_account.Credentials.from_service_account_info(creds_dict)
else:
    creds = None


OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
QDRANT_URL = os.environ["QDRANT_URL"]
COLLECTION_NAME = os.environ["COLLECTION"]


rag_prompt_template = """Sei un assistente accademico.
Hai accesso a estratti di documenti ufficiali.

Usa SOLO il contesto fornito per rispondere.
Se il contesto contiene il termine o l'argomento richiesto, indica che è presente e copia il testo più rilevante.
Non aggiungere nulla di tuo e non inventare.

Se davvero non ci sono riferimenti nemmeno parziali, rispondi esattamente: "Non presente nei documenti".

Domanda: {question}

Contesto:
{context}

Risposta:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=rag_prompt_template,
)

classifier_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Sei un classificatore di query.

Se la domanda riguarda saluti, domande generiche, curiosità non legate a regolamenti o procedure accademiche → rispondi SOLO con: semplice

Se la domanda richiede informazioni ufficiali su corsi, tesi, tirocini, lauree, esami, regolamenti, scadenze → rispondi SOLO con: rag

Domanda: {question}
Categoria:""",
)

simple_prompt_template = """Sei un assistente accademico gentile.
Rispondi in modo breve e diretto alla domanda generica seguente:

Domanda: {question}
Risposta:"""


embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)

print("Connettendo a Qdrant...")
qdrant_client = QdrantClient(url=QDRANT_URL)

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
)
print("Connesso al vector store con successo!")

print("Inizializzando LLM...")

# llama locale:
# llm = OllamaLLM(model="llama3.2:3b", base_url=OLLAMA_BASE_URL)

# gemini:
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
)

# llama 3.3 70b api:
#llm = ChatVertexAI(
#    model="llama-3.3-70b-instruct-maas",
#    location="us-central1",
#    temperature=0,
#    max_output_tokens=1024,
#    credentials=creds,
#)


all_texts = []
scroll_filter = None
while True:
    points, next_page = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        limit=1000, 
        offset=scroll_filter
    )
    for p in points:
        text = p.payload.get("page_content", "")
        if text:
            all_texts.append(text)
    if next_page is None:
        break
    scroll_filter = next_page


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)

tokenized_corpus = [word_tokenize(text.lower()) for text in all_texts]
bm25 = BM25Okapi(tokenized_corpus)


def tfidf_search(query: str, k: int = 5):
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix @ query_vec.T).toarray().ravel()
    top_indices = np.argsort(scores)[::-1][:k]
    results = [(all_texts[i], scores[i]) for i in top_indices]
    return results

def bm25_search(query: str, k: int = 5):
    """Ricerca sparse basata su BM25"""
    query_tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    results = [(all_texts[i], scores[i]) for i in top_indices]
    return results


def classify_query(query: str) -> str:
    """Classifica la query in 'semplice' o 'rag'"""
    try:
        classification = llm.invoke(classifier_prompt.format(question=query))
        if hasattr(classification, "content"):
            classification = classification.content
        classification = str(classification).strip().lower()
        if "semplice" in classification:
            return "semplice"
        else:
            return "rag"
    except Exception as e:
        print(f"Errore classificazione: {e}")
        return "rag"

def answer_query_dense(query: str):
    """Usa solo Dense (Qdrant)"""
    vec = embeddings.embed_query(query)
    docs = vectorstore.similarity_search_by_vector(vec, k=8)

    seen = set()
    unique_docs = []
    for d in docs:
        text = d.page_content.strip()
        if text not in seen:
            seen.add(text)
            unique_docs.append(d)
        if len(unique_docs) == 5:
            break

    if not unique_docs:
        return "Non presente nei documenti"

    context = "\n\n".join(
        [f"[Fonte {i+1}] ({doc.metadata.get('source_url', 'N/A')})\n{doc.page_content}"
         for i, doc in enumerate(unique_docs)]
    )

    prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
"""
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content

    return f"Risposta Dense: {answer}"

def answer_query_tfidf(query: str):
    """Usa solo Sparse (TF-IDF)"""
    tfidf_results = tfidf_search(query, k=5)
    if not tfidf_results:
        return "Non presente nei documenti"

    context = "\n\n".join(
        [f"[Fonte {i+1}] (TF-IDF, score={score:.3f})\n{text}"
         for i, (text, score) in enumerate(tfidf_results)]
    )

    prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
"""
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content

    return f"Risposta TF-IDF: {answer}"

def answer_query_bm25(query: str):
    """Usa BM25 al posto di TF-IDF"""
    bm25_results = bm25_search(query, k=5)
    if not bm25_results:
        return "Non presente nei documenti"

    context = "\n\n".join(
        [f"[Fonte {i+1}] (BM25, score={score:.3f})\n{text}"
         for i, (text, score) in enumerate(bm25_results)]
    )

    prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
"""
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content

    return f"Risposta BM25: {answer}"



def compare_dense_vs_tfidf(query: str):
    """Confronta Dense vs Sparse"""
    print(f"\n🔎 Query: {query}\n{'='*70}")

    dense_answer = answer_query_dense(query)
    print("\n--- 📘 Risposta Dense (Qdrant) ---")
    print(dense_answer)

    sparse_answer = answer_query_tfidf(query)
    print("\n--- 📗 Risposta Sparse (TF-IDF) ---")
    print(sparse_answer)

    print("="*70)

def hybrid_search(query: str, alpha: float = 0.5, k: int = 5):
    """
    Esegue retrieval ibrido combinando Dense (Qdrant) e Sparse (TF-IDF).
    alpha bilancia i due contributi: 
      - alpha=1 usa solo Dense
      - alpha=0 usa solo TF-IDF
    """
    # --- Dense retrieval ---
    dense_vec = embeddings.embed_query(query)
    dense_docs = vectorstore.similarity_search_with_score_by_vector(dense_vec, k=10)
    dense_results = {}
    for doc, score in dense_docs:
        dense_results[doc.page_content] = 1 - score  # Qdrant restituisce distanza, la inverto

    # --- Sparse retrieval ---
    #tfidf_results = tfidf_search(query, k=10)
    tfidf_results = bm25_search(query, k=10)
    sparse_results = {text: score for text, score in tfidf_results}

    # --- Normalizzazione punteggi ---
    all_texts = set(dense_results.keys()) | set(sparse_results.keys())
    dense_scores = np.array([dense_results.get(t, 0) for t in all_texts])
    sparse_scores = np.array([sparse_results.get(t, 0) for t in all_texts])

    if dense_scores.max() > 0:
        dense_scores = dense_scores / dense_scores.max()
    if sparse_scores.max() > 0:
        sparse_scores = sparse_scores / sparse_scores.max()

    # --- Fusione pesata ---
    hybrid_scores = alpha * dense_scores + (1 - alpha) * sparse_scores

    # --- Ordina per punteggio combinato ---
    sorted_indices = np.argsort(hybrid_scores)[::-1][:k]
    top_texts = [list(all_texts)[i] for i in sorted_indices]
    top_scores = hybrid_scores[sorted_indices]

    # --- Prepara il contesto per l’LLM ---
    context = "\n\n".join(
        [f"[Fonte {i+1}] (Hybrid score={s:.3f})\n{t}" for i, (t, s) in enumerate(zip(top_texts, top_scores))]
    )

    prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
"""
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content

    return f"Risposta ibrida (α={alpha}): {answer}"

    

# def answer_query(query: str):
#     try:
#         # STEP 1: classifica la query
#         mode = classify_query(query)
#         print(f"Classificazione query: {mode}")

#         # STEP 2: risposta semplice
#         if mode == "semplice":
#             prompt = simple_prompt_template.format(question=query)
#             answer = llm.invoke(prompt)
#             if hasattr(answer, "content"):
#                 answer = answer.content
#             return f"Risposta semplice: {answer}"

#         # STEP 3: risposta RAG
#         print("Processando query con RAG...")
#         vec = embeddings.embed_query(query)
#         docs = vectorstore.similarity_search_by_vector(vec, k=8)

#         seen = set()
#         unique_docs = []
#         for d in docs:
#             text = d.page_content.strip()
#             if text not in seen:
#                 seen.add(text)
#                 unique_docs.append(d)
#             if len(unique_docs) == 5:
#                 break

#         if not unique_docs:
#             return "Non presente nei documenti"

#         context = "\n\n".join(
#             [f"[Fonte {i+1}] ({doc.metadata.get('source_url', 'N/A')})\n{doc.page_content}"
#              for i, doc in enumerate(unique_docs)]
#         )

#         prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

#     Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
#     """
#         answer = llm.invoke(prompt)
#         if hasattr(answer, "content"):
#             answer = answer.content

#         main_source = unique_docs[0].metadata.get("source_url", "N/A")
#         response = f"Risposta: {answer}\n"
#         response += f"\nPer ulteriori informazioni consulta il seguente link: {main_source}\n"

#         if unique_docs:
#             response += f"\nFonti consultate ({len(unique_docs)} documenti):"
#             for i, doc in enumerate(unique_docs, 1):
#                 preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
#                 response += f"\n{i}. {preview}"
#         return response

#     except Exception as e:
#         return f"Errore durante la query: {e}"


def test_connection():
    try:
        vec = embeddings.embed_query("test")
        docs = vectorstore.max_marginal_relevance_search_by_vector(vec, k=3, fetch_k=10, lambda_mult=0.5)
        print(f"Test connessione riuscito.")
        for i, doc in enumerate(docs, 1):
            preview = doc.page_content[:120].replace("\n", " ")
        return True
    except Exception as e:
        print(f"Test connessione fallito: {e}")
        return False


if __name__ == "__main__":
    print("Sistema di Q&A avviato.")

    if not test_connection():
        print("Impossibile connettersi al vector store. Verificare che la collezione esista.")
        exit(1)

    print("Connessione verificata. Digita 'exit' o 'quit' per uscire.\n")

    while True:
        try:
            q = input("Domanda: ")
            if q.lower() in ["exit", "quit"]:
                break
            if q.strip():
                mode = classify_query(q)
                if mode == "semplice":
                    prompt = simple_prompt_template.format(question=q)
                    answer = llm.invoke(prompt)
                    if hasattr(answer, "content"):
                        answer = answer.content
                    print(f"\nRisposta semplice: {answer}\n")
                else:
                    # confronto automatico Dense vs Sparse
                    compare_dense_vs_tfidf(q)
                    print("\n--- 📙 Risposta Ibrida (Dense + TF-IDF) ---")
                    hybrid_answer = hybrid_search(q, alpha=0.6, k=5)
                    print(hybrid_answer)

                print("-" * 50)
            else:
                print("Inserisci una domanda valida.")
        except KeyboardInterrupt:
            print("\nUscita...")
            break
        except Exception as e:
            print(f"Errore: {e}")



    # while True:
    #     try:
    #         q = input("Domanda: ")
    #         if q.lower() in ["exit", "quit"]:
    #             break
    #         if q.strip():
    #             result = answer_query(q)
    #             print(f"\n{result}\n")
    #             print("-" * 50)
    #         else:
    #             print("Inserisci una domanda valida.")
    #     except KeyboardInterrupt:
    #         print("\nUscita...")
    #         break
    #     except Exception as e:
    #         print(f"Errore: {e}")