#!/bin/bash

CONTAINER="indexing"
FILE="indexing/links.txt"

while IFS= read -r line; do
    # Rimuovi spazi iniziali/finali
    line=$(echo "$line" | xargs)

    # Salta righe vuote o commenti
    if [[ -z "$line" || "$line" =~ ^# ]]; then
        continue
    fi

    # La prima parola della riga è l'URL
    url=$(echo "$line" | awk '{print $1}')

    # Le parole successive sono argomenti extra (se esistono)
    extra_args=$(echo "$line" | cut -s -d' ' -f2-)

    echo ">>> Processando: $url $extra_args"

    if [[ -n "$extra_args" ]]; then
        docker exec "$CONTAINER" python indexing.py --url "$url" $extra_args
    else
        docker exec "$CONTAINER" python indexing.py --url "$url"
    fi
done < "$FILE"

echo ">>> Processo completato."