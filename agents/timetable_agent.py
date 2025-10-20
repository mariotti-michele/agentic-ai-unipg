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

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = "ing_info_mag_docs"

prompt_template = """Sei un assistente specializzato nella gestione degli orari dei corsi
del Corso di Laurea Magistrale in Ingegneria Informatica e Robotica.

Hai nella tua conoscenza le tabelle degli orari delle lezioni in formato JSON strutturato con questa forma:

{{
  "corso_di_laurea": "...",
  "anno_accademico": "...",
  "anno": "...",
  "semestre": "...",
  "periodo": "...",
  "orario": {{
    "Giorno": {{
      "OraInizio-OraFine": [[
        {{ "corso": "...", "aula": "..." }}
      ]]
    }}
  }}
}}

Nota:
In alcuni rari casi nello stesso slot orario possono comparire più corsi in parallelo svolti in aule diverse. Ad esempio:
"lunedì": {{
      "09:00-11:00": [[
        {{ "corso": "Corso A Corso B", "aula": "A1 B2" }},
      ]]
    }}
Li puoi riconoscere perché il campo "corso" contiene più nomi di corsi separati solo da spazi (senza congiunzioni) e il campo "aula" contiene più aule.
In questo caso considera ciascun corso con la sua aula come elemento distinto.

Rispondi alle domande relative a giorni, orari, corsi e aule in base ai dati forniti.

Usa SOLO il contesto fornito, senza aggiungere informazioni esterne.
Se non trovi riferimenti, rispondi esattamente: "Non presente nei documenti".

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

import argparse

parser = argparse.ArgumentParser(description="Sistema Q&A con modelli selezionabili")
parser.add_argument("--model", type=str, default="llama-api",
                    choices=["llama-local", "gemini", "llama-api"],
                    help="Seleziona il modello da usare")
args = parser.parse_args()

print(f"Inizializzando LLM con modello: {args.model}")

if args.model == "llama-local":
    # llama locale
    llm = OllamaLLM(model="llama3.2:3b", base_url=OLLAMA_BASE_URL)

elif args.model == "gemini":
    # gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )

elif args.model == "llama-api":
    # llama 3.3 70b API (Vertex AI)
    llm = ChatVertexAI(
        model="llama-3.3-70b-instruct-maas",
        location="us-central1",
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
        answer = llm.invoke(prompt)
        if hasattr(answer, "content"):
            answer = answer.content


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