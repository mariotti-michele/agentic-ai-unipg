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

if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
    creds_dict = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    creds = service_account.Credentials.from_service_account_info(creds_dict)
else:
    creds = None

from langchain_google_vertexai import ChatVertexAI


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = "ing_info_mag_docs"

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


QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

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
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0.2,
# )

# llama 3.3 70b api:
llm = ChatVertexAI(
    model="llama-3.3-70b-instruct-maas",
    temperature=0,
    max_output_tokens=1024,
    credentials=creds,
)

print("Creando retriever...")

def answer_query(query: str):
    try:
        print(f"Processando query...")

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
        # llama locale e llama 3.3 70b api:
        answer = llm.invoke(prompt)

        # gemini:
        # answer = llm.invoke(prompt).content


        main_source = unique_docs[0].metadata.get("source_url", "N/A")

        response = f"Risposta: {answer}\n"
        response += f"\nPer ulteriori informazioni consulta il seguente link: {main_source}\n"

        if unique_docs:
            response += f"\nFonti consultate ({len(unique_docs)} documenti):"
            for i, doc in enumerate(unique_docs, 1):
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                response += f"\n{i}. {preview}"
        return response

    except Exception as e:
        return f"Errore durante la query: {e}"


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