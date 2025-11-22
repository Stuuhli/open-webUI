import chromadb
from chromadb.config import Settings
import os

# Pfad zur DB im Container (oder lokal gemountet)
# Wenn du das Skript AUSSERHALB von Docker ausführst, muss der Pfad stimmen
# (z.B. 'backend/data/vector_db')
DB_PATH = "./data/vector_db" 

try:
    client = chromadb.PersistentClient(path=DB_PATH)
    
    print("Verfügbare Collections:")
    collections = client.list_collections()
    for c in collections:
        print(f" - {c.name}")
        
    if collections:
        # Nimm die letzte (oder eine spezifische 'file-...')
        col_name = collections[-1].name 
        print(f"\nUntersuche Collection: {col_name}")
        
        collection = client.get_collection(col_name)
        
        # Hole die ersten 3 Dokumente
        results = collection.get(limit=3)
        
        for i, doc in enumerate(results['documents']):
            print(f"\n--- CHUNK {i} ---")
            print(f"ID: {results['ids'][i]}")
            print(f"METADATA: {results['metadatas'][i]}")
            print(f"CONTENT (ersten 300 Zeichen):\n{doc[:300]}...")
            print("-----------------")

except Exception as e:
    print(f"Fehler: {e}")
    print("Hinweis: Stelle sicher, dass der Pfad stimmt und keine andere Instanz die DB lockt.")