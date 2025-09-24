import os
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
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



# Inizializza gli embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)

try:
    # Metodo 1: Approccio semplificato
    vectorstore = QdrantVectorStore.from_existing_collection(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,  # Nota: 'embedding' non 'embeddings'
        url=QDRANT_URL,
    )
    print("Connesso al vector store con successo!")
    
except Exception as e:
    print(f"Errore metodo 1: {e}")
    
    try:
        # Metodo 2: Creazione manuale del client
        qdrant_client = QdrantClient(url=QDRANT_URL)
        
        vectorstore = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
        print("Connesso al vector store con il metodo 2!")
        
    except Exception as e2:
        print(f"Errore metodo 2: {e2}")
        
        try:
            # Metodo 3: Senza parametro embedding (se la collezione esiste già)
            vectorstore = QdrantVectorStore.from_existing_collection(
                collection_name=COLLECTION_NAME,
                url=QDRANT_URL,
            )
            # Assegna manualmente gli embeddings
            vectorstore.embeddings = embeddings
            print("Connesso al vector store con il metodo 3!")
            
        except Exception as e3:
            print(f"Errore metodo 3: {e3}")
            
            # Metodo 4: Ultimo tentativo con client separato senza embedding
            try:
                qdrant_client = QdrantClient(url=QDRANT_URL)
                vectorstore = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=COLLECTION_NAME,
                )
                # Assegna manualmente gli embeddings
                vectorstore.embeddings = embeddings
                print("Connesso al vector store con il metodo 4!")
                
            except Exception as e4:
                print(f"Tutti i metodi falliti. Ultimo errore: {e4}")
                exit(1)

# Verifica che il vectorstore sia stato creato
if 'vectorstore' not in locals():
    print("Impossibile creare il vector store")
    exit(1)

# 2. Inizializza Ollama come LLM
print("Inizializzando LLM...")
llm = OllamaLLM(model="llama3.1:8b", base_url=OLLAMA_BASE_URL)

# 3. Crea il retriever
print("Creando retriever...")
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 4. RetrievalQA
print("Inizializzando RetrievalQA...")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)

def answer_query(query: str):
    try:
        print(f"Processando query: {query}")
        result = qa({"query": query})
        
        # Estrai la risposta e le fonti
        answer = result.get("result", "Nessuna risposta trovata")
        sources = result.get("source_documents", [])
        
        response = f"Risposta: {answer}\n"
        if sources:
            response += f"\nFonti consultate ({len(sources)} documenti):"
            for i, doc in enumerate(sources[:3], 1):  # Mostra solo le prime 3 fonti
                # Estrai un breve estratto del contenuto
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                response += f"\n{i}. {content_preview}"
        
        return response
        
    except Exception as e:
        return f"Errore durante la query: {e}"

def test_connection():
    """Test la connessione al vector store"""
    try:
        # Test semplice di ricerca
        docs = vectorstore.similarity_search("test", k=1)
        print(f"Test connessione riuscito. Trovati {len(docs)} documenti.")
        return True
    except Exception as e:
        print(f"Test connessione fallito: {e}")
        return False

if __name__ == "__main__":
    print("Sistema di Q&A avviato.")
    
    # Test della connessione
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