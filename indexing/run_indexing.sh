#!/bin/bash

CONTAINER="indexing"
QDRANT_URL="http://localhost:6333"

echo ">>> Resetto le collezioni Qdrant..."
for c in $(curl -s "$QDRANT_URL/collections" | grep -o '"name":"[^"]*' | sed 's/"name":"//'); do
  curl -s -X DELETE "$QDRANT_URL/collections/$c" >/dev/null
done
echo ">>> Operazione completata."

echo ""
echo ">>> Avvio indicizzazione multipla da links.txt ..."
docker exec "$CONTAINER" python indexing.py
echo ">>> Indicizzazione completata."
