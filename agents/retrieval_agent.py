import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------------- Config ----------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "ing_info_mag_docs"

prompt_template = """Sei un assistente accademico.
Usa solo i documenti forniti nel contesto per rispondere.
Se il contesto contiene l'informazione, riportala in modo preciso (anche copiandola).
Se il contesto non contiene l'informazione, rispondi solo: "Non presente nei documenti".

Domanda: {question}

Contesto:
{context}

Risposta:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# ---------------- Embeddings ----------------
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)

vec = embeddings.embed_query("test")
print(f"[DEBUG] Dimensione embedding Ollama: {len(vec)}")

# ---------------- Qdrant ----------------
print("Connettendo a Qdrant...")
qdrant_client = QdrantClient(url=QDRANT_URL)

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
)
print("Connesso al vector store con successo!")

# ---------------- LLM ----------------
print("Inizializzando LLM...")
llm = OllamaLLM(model="llama3.2:3b", base_url=OLLAMA_BASE_URL)

# ---------------- Retriever + QA ----------------
print("Creando retriever...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print("Inizializzando RetrievalQA...")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)

# ---------------- Funzioni ----------------
def answer_query(query: str):
    try:
        print(f"Processando query: {query}")
        vec = embeddings.embed_query(query)
        docs = vectorstore.similarity_search_by_vector(vec, k=5)

        context = "\n".join([d.page_content for d in docs])

        prompt = QA_CHAIN_PROMPT.format(context=context, question=query)
        answer = llm.invoke(prompt)

        response = f"Risposta: {answer}\n"
        if docs:
            response += f"\nFonti consultate ({len(docs)} documenti):"
            for i, doc in enumerate(docs[:3], 1):
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                response += f"\n{i}. {preview}"
        return response

    except Exception as e:
        return f"Errore durante la query: {e}"


def test_connection():
    """Test connessione al vector store con query esplicita"""
    try:
        vec = embeddings.embed_query("test")
        docs = vectorstore.similarity_search_by_vector(vec, k=1)
        print(f"Test connessione riuscito. Trovati {len(docs)} documenti.")
        return True
    except Exception as e:
        print(f"Test connessione fallito: {e}")
        return False

# ---------------- Main ----------------
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
                result = answer_query(q)
                print(f"\n{result}\n")
                print("-" * 50)
            else:
                print("Inserisci una domanda valida.")
        except KeyboardInterrupt:
            print("\nUscita...")
            break
        except Exception as e:
            print(f"Errore: {e}")
