import tkinter as tk
from tkinter import filedialog
import csv
import json
import os
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Ostrzeżenie: Biblioteka NumPy nie jest zainstalowana. Opcja zapisu do .npy będzie niedostępna.")

# Definicja tokenów specjalnych
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>" # Dla nieznanych k-gramów
SOS_TOKEN = "<SOS>" # Start of Sequence
EOS_TOKEN = "<EOS>" # End of Sequence

# Lista standardowych aminokwasów (może być używana do walidacji itp., ale nie jest bezpośrednio tokenami)
AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]

def generate_k_grams(sequence_string, k):
    """Generuje listę k-gramów z danej sekwencji (stringa)."""
    # Zakładamy, że sequence_string jest już przetworzony (np. wielkie litery)
    if not isinstance(sequence_string, str):
        return [] # Oczekujemy stringa
    n = len(sequence_string)
    if k <= 0 or n < k:
        return [] # Zwraca pustą listę, jeśli k jest niepoprawne lub sekwencja za krótka
    return [sequence_string[i:i+k] for i in range(n - k + 1)]

def tokenize_sequence_kgrams(processed_sequence_string, k, token_to_id_map):
    """Tokenizuje pojedynczą sekwencję na k-gramy i specjalne tokeny, zwracając listę ID."""
    k_grams = generate_k_grams(processed_sequence_string, k)
    
    # Pobierz ID tokenów specjalnych ze słownika
    sos_id = token_to_id_map[SOS_TOKEN]
    eos_id = token_to_id_map[EOS_TOKEN]
    unk_id = token_to_id_map[UNK_TOKEN]
    
    token_ids = [sos_id]
    for k_gram in k_grams:
        token_ids.append(token_to_id_map.get(k_gram, unk_id))
    token_ids.append(eos_id)
    return token_ids

def ask_output_format_console():
    """Pyta użytkownika o format wyjściowy poprzez konsolę."""
    print("\nWybierz format pliku wyjściowego:")
    options = {
        "a": ("JSON (.json)", "json"),
        "b": ("CSV (.csv + _vocab.json)", "csv")
    }
    if NUMPY_AVAILABLE:
        options["c"] = ("NumPy (.npy + _vocab.json)", "npy")

    for key, (description, _) in options.items():
        print(f"  {key}) {description}")

    while True:
        choice = input(f"Wybierz opcję ({', '.join(options.keys())}): ").lower().strip()
        if choice in options:
            return options[choice][1]
        else:
            print("Nieprawidłowy wybór, spróbuj ponownie.")

def main():
    root = tk.Tk()
    root.withdraw()

    # Pobierz wartość k od użytkownika
    while True:
        try:
            k_value_str = input("Podaj długość k-gramu (np. 3 dla trypletów): ").strip()
            k_value = int(k_value_str)
            if k_value <= 0:
                print("Wartość k musi być liczbą dodatnią.")
            else:
                break
        except ValueError:
            print("Nieprawidłowa wartość. Proszę podać liczbę całkowitą.")
    print(f"Używana długość k-gramu: {k_value}")

    # Wybór pliku wejściowego
    print("DEBUG: Tymczasowe uaktywnienie root dla askopenfilename.")
    root.deiconify()
    root.update()
    root.update_idletasks()
    input_filepath = filedialog.askopenfilename(
        master=root, 
        title="Wybierz plik .csv z sekwencjami aminokwasów",
        filetypes=(("Pliki CSV", "*.csv"), ("Wszystkie pliki", "*.*"))
    )
    print("DEBUG: Ponowne ukrycie root po askopenfilename.")
    root.withdraw()
    root.update()

    if not input_filepath:
        print("Nie wybrano pliku wejściowego. Zamykanie programu.")
        root.destroy()
        return
    print(f"Wybrano plik wejściowy: {input_filepath}")

    # Wczytywanie sekwencji
    raw_sequences = []
    try:
        with open(input_filepath, mode='r', encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row: 
                    sequence_str = row[0].strip().upper() # Przetwarzanie od razu na wielkie litery
                    if sequence_str: 
                        raw_sequences.append(sequence_str)
    except FileNotFoundError:
        print(f"Błąd: Plik {input_filepath} nie został znaleziony.")
        root.destroy()
        return
    except Exception as e:
        print(f"Wystąpił błąd podczas odczytu pliku CSV: {e}")
        root.destroy()
        return

    if not raw_sequences:
        print("Nie znaleziono żadnych sekwencji w pliku. Zamykanie programu.")
        root.destroy()
        return
    print(f"Znaleziono {len(raw_sequences)} sekwencji.")

    # Budowanie słownika
    print("Budowanie słownika tokenów (k-gramów i specjalnych)...")
    token_to_id = {}
    id_counter = 0
    def add_token_to_vocab(token, vocab_dict_ref): # Zmieniono nazwę argumentu
        nonlocal id_counter # odnosi się do id_counter z main()
        if token not in vocab_dict_ref:
            vocab_dict_ref[token] = id_counter
            id_counter += 1

    # Dodaj tokeny specjalne jako pierwsze
    add_token_to_vocab(PAD_TOKEN, token_to_id)
    add_token_to_vocab(UNK_TOKEN, token_to_id)
    add_token_to_vocab(SOS_TOKEN, token_to_id)
    add_token_to_vocab(EOS_TOKEN, token_to_id)

    # Zbierz wszystkie unikalne k-gramy z danych
    all_k_grams_set = set()
    for seq_str in raw_sequences:
        # seq_str jest już uppercased podczas wczytywania
        generated_k_grams_for_seq = generate_k_grams(seq_str, k_value)
        all_k_grams_set.update(generated_k_grams_for_seq)
    
    # Dodaj k-gramy do słownika (posortowane dla spójności)
    for k_gram in sorted(list(all_k_grams_set)):
        add_token_to_vocab(k_gram, token_to_id)
    
    vocabulary = token_to_id
    print(f"Słownik zbudowany. Rozmiar: {len(vocabulary)} tokenów.")

    # Tokenizacja wszystkich sekwencji
    print("Tokenizowanie sekwencji...")
    tokenized_sequences_ids = []
    for seq_str in raw_sequences:
        # seq_str jest już uppercased
        ids = tokenize_sequence_kgrams(seq_str, k_value, vocabulary)
        tokenized_sequences_ids.append(ids)

    # Padding
    if not tokenized_sequences_ids:
        print("Brak stokenizowanych sekwencji do przetworzenia (np. wszystkie były za krótkie dla k-gramów).")
        root.destroy()
        return

    max_len = 0
    if tokenized_sequences_ids: # Upewnij się, że lista nie jest pusta
         max_len = max(len(ts) for ts in tokenized_sequences_ids) if tokenized_sequences_ids else 0
    
    print(f"Maksymalna długość stokenizowanej sekwencji (z <SOS>/<EOS>): {max_len}")

    padded_sequences = []
    pad_id = vocabulary[PAD_TOKEN]
    for ts_ids in tokenized_sequences_ids:
        padding_needed = max_len - len(ts_ids)
        padded_seq = ts_ids + ([pad_id] * padding_needed)
        padded_sequences.append(padded_seq)
    print("Tokenizacja i padding zakończone.")

    # Wybór formatu i zapis
    output_format = ask_output_format_console()
    if not output_format: 
        print("Nie wybrano formatu zapisu. Anulowano.")
        root.destroy()
        return
    print(f"Wybrano format zapisu: {output_format.upper()}")

    print("DEBUG: Tymczasowe uaktywnienie root dla asksaveasfilename.")
    root.deiconify()
    root.update()
    root.update_idletasks()
    print("DEBUG: Po deiconify/update, tuż przed filedialog.asksaveasfilename")

    file_extensions_map = {
        "json": (("Pliki JSON", "*.json"),),
        "csv": (("Pliki CSV", "*.csv"),),
        "npy": (("Pliki NumPy", "*.npy"),)
    }
    current_default_ext = f".{output_format}"
    dialog_defaultextension = "" if output_format in ["csv", "npy"] else current_default_ext
        
    output_filepath_base = filedialog.asksaveasfilename(
        master=root,
        title=f"Zapisz stokenizowane sekwencje jako ({output_format.upper()})...",
        defaultextension=dialog_defaultextension, 
        filetypes=file_extensions_map.get(output_format, (("Wszystkie pliki", "*.*"),))
    )
    print(f"DEBUG: Po filedialog.asksaveasfilename, wartość output_filepath_base: '{output_filepath_base}'")
    print("DEBUG: Ponowne ukrycie root po asksaveasfilename.")
    root.withdraw()
    root.update()

    if not output_filepath_base:
        print("Nie wybrano lokalizacji zapisu. Anulowano.")
        root.destroy()
        return

    try:
        file_name, ext = os.path.splitext(output_filepath_base)
        if not ext.lower() == current_default_ext:
            output_filepath_with_ext = file_name + current_default_ext
        else: 
            output_filepath_with_ext = output_filepath_base
        if output_format in ["csv", "npy"]:
            if not os.path.splitext(output_filepath_with_ext)[1]:
                 output_filepath_with_ext += current_default_ext
            elif output_filepath_with_ext.endswith("."):
                 output_filepath_with_ext = output_filepath_with_ext[:-1] + current_default_ext

        if output_format == "json":
            output_data = {
                "vocabulary": vocabulary, # Zapisujemy pełny słownik k-gramów
                "k_value": k_value,     # Dodajemy informację o użytym k
                "tokenized_sequences": padded_sequences
            }
            with open(output_filepath_with_ext, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            print(f"Stokenizowane dane zostały pomyślnie zapisane do: {output_filepath_with_ext}")

        elif output_format == "csv":
            sequences_csv_path = output_filepath_with_ext
            with open(sequences_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(padded_sequences)
            print(f"Stokenizowane sekwencje zapisano do: {sequences_csv_path}")

            vocab_data_for_csv_npy = {"vocabulary": vocabulary, "k_value": k_value}
            vocab_json_path = os.path.splitext(sequences_csv_path)[0] + "_vocab.json"
            with open(vocab_json_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data_for_csv_npy, f, indent=4)
            print(f"Słownik i wartość k zapisano do: {vocab_json_path}")

        elif output_format == "npy" and NUMPY_AVAILABLE:
            sequences_npy_path = output_filepath_with_ext
            np_array = np.array(padded_sequences, dtype=np.int32) 
            np.save(sequences_npy_path, np_array)
            print(f"Stokenizowane sekwencje zapisano do: {sequences_npy_path}")
            
            vocab_data_for_csv_npy = {"vocabulary": vocabulary, "k_value": k_value}
            vocab_json_path = os.path.splitext(sequences_npy_path)[0] + "_vocab.json"
            with open(vocab_json_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data_for_csv_npy, f, indent=4)
            print(f"Słownik i wartość k zapisano do: {vocab_json_path}")
        
        elif output_format == "npy" and not NUMPY_AVAILABLE:
            print("Błąd: Nie można zapisać w formacie NPY, ponieważ biblioteka NumPy nie jest dostępna.")

    except Exception as e:
        print(f"Wystąpił błąd podczas zapisu pliku: {e}")
    finally:
        print("Zamykanie aplikacji.")
        root.destroy()

if __name__ == "__main__":
    main()
