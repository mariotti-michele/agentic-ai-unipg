def calcola_formula(prezzo_input, prezzo_output):
    #risultato = (250 * prezzo_input + 75 * prezzo_output) * 130 / 1000000
    risultato = (1600 * prezzo_input + 150 * prezzo_output) / 1000000
    return risultato

try:
    prezzo_input = float(input("Inserisci il valore di prezzo_input: "))
    prezzo_output = float(input("Inserisci il valore di prezzo_output: "))

    risultato = calcola_formula(prezzo_input, prezzo_output)
    print(f"Il risultato della formula è: {risultato}")
except ValueError:
    print("Errore.")
