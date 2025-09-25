import os
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = "ing_info_mag_docs"

def diagnose_qdrant():
    """Diagnostica dettagliata dei problemi Qdrant"""
    
    print("=== DIAGNOSTICA QDRANT ===")
    
    try:
        print("1. Tentativo connessione a Qdrant...")
        client = QdrantClient(url=QDRANT_URL)
        print("✓ Connessione riuscita")
        
        print("\n2. Lista collezioni...")
        collections = client.get_collections()
        print(f"✓ Collezioni trovate: {[c.name for c in collections.collections]}")
        
        if COLLECTION_NAME not in [c.name for c in collections.collections]:
            print(f"✗ Collezione '{COLLECTION_NAME}' NON TROVATA!")
            return False
            
        print(f"\n3. Test lettura collezione '{COLLECTION_NAME}'...")
        try:
            info = client.get_collection(COLLECTION_NAME)
            print(f"✓ Collezione letta con successo")
            print(f"  - Punti totali: {info.points_count}")
            print(f"  - Dimensione vettori: {info.config.params.vectors}")
            
        except Exception as e:
            print(f"✗ ERRORE lettura collezione: {e}")
            print("Questo è probabilmente il problema!")
            return False
            
        print(f"\n4. Test query semplice...")
        try:
            # Test di ricerca senza vettore (solo metadati)
            result = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1
            )
            print(f"✓ Query riuscita, documenti trovati: {len(result[0])}")
            
        except Exception as e:
            print(f"✗ ERRORE query: {e}")
            return False
            
        print(f"\n5. Test ricerca similarity (può richiedere embedding)...")
        try:
            # Questo potrebbe fallire se ci sono problemi con i vettori
            result = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=[0.1] * 768,  # Vettore dummy
                limit=1
            )
            print(f"✓ Ricerca similarity riuscita")
            
        except Exception as e:
            print(f"✗ ERRORE ricerca similarity: {e}")
            print("Possibile problema con i vettori memorizzati")
            return False
            
        print("\n✅ TUTTI I TEST PASSATI - Qdrant funziona correttamente")
        return True
        
    except Exception as e:
        print(f"✗ ERRORE CRITICO: {e}")
        return False

def fix_collection():
    """Tenta di riparare o ricreare la collezione"""
    
    print("\n=== TENTATIVO RIPARAZIONE ===")
    
    try:
        client = QdrantClient(url=QDRANT_URL)
        
        print("1. Backup informazioni collezione...")
        try:
            info = client.get_collection(COLLECTION_NAME)
            vector_config = info.config.params.vectors
            print(f"Config vettori salvata: {vector_config}")
        except:
            print("Impossibile leggere config - useremo default")
            vector_config = None
        
        print("2. Eliminazione collezione corrotta...")
        client.delete_collection(COLLECTION_NAME)
        print("✓ Collezione eliminata")
        
        print("3. Ricreazione collezione...")
        from qdrant_client.http.models import Distance, VectorParams
        
        # Usa la config salvata o default
        if vector_config:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=vector_config
            )
        else:
            # Default per nomic-embed-text (768 dimensioni)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        
        print("✓ Collezione ricreata")
        print("⚠️  ATTENZIONE: Dovrai re-indicizzare i tuoi documenti!")
        
        return True
        
    except Exception as e:
        print(f"✗ Errore durante riparazione: {e}")
        return False

if __name__ == "__main__":
    print("Avvio diagnostica Qdrant...")
    
    if diagnose_qdrant():
        print("\n🎉 Qdrant funziona correttamente!")
    else:
        print("\n💥 Problemi rilevati!")
        
        risposta = input("\nVuoi tentare la riparazione? (eliminerà tutti i dati) [s/N]: ")
        if risposta.lower() == 's':
            if fix_collection():
                print("\n✅ Riparazione completata!")
                print("Ora dovrai re-eseguire l'indicizzazione dei documenti.")
            else:
                print("\n❌ Riparazione fallita.")
                print("Prova a riavviare completamente Qdrant.")
        else:
            print("\nOperazione annullata.")
            print("Suggerimenti:")
            print("1. Riavvia il container Qdrant")
            print("2. Controlla i log: docker logs qdrant")
            print("3. Se il problema persiste, elimina il volume Qdrant")