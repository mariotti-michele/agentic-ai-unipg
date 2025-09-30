#!/bin/bash

CONTAINER="indexing"

FILE="indexing/links.txt"

while IFS= read -r line; do
    # Salta righe vuote o commenti
    [[ -z "$line" || "$line" =~ ^# ]] && continue

    # La prima parola della riga è l'URL
    url=$(echo "$line" | awk '{print $1}')
    echo "$url"

    # Le parole successive sono argomenti extra
    extra_args=$(echo "$line" | cut -d' ' -f2-)
    echo "$extra_args"

    echo ">>> Processando: $url $extra_args"
    docker exec "$CONTAINER" python indexing.py --url "$url" $extra_args
done < "$FILE"
echo ">>> Processo completato."