import json
import os
from collections import defaultdict
import numpy as np
import tkinter as tk
from tkinter import filedialog

# --- Implementacja Byte Pair Encoding (BPE) ---

def get_stats(vocab_sequences):
    """
    Oblicza częstotliwość występowania par sąsiadujących tokenów (symboli)
    w korpusie sekwencji (każda sekwencja to lista tokenów-stringów).
    """
    pairs = defaultdict(int)
    for seq in vocab_sequences:
        for i in range(len(seq) - 1):
            pairs[(seq[i], seq[i+1])] += 1
    return pairs

def merge_vocab(pair_to_merge, vocab_sequences):
    """
    Łączy daną parę tokenów w nowy, pojedynczy token we wszystkich sekwencjach korpusu.
    Zwraca nowy korpus sekwencji.
    """
    new_token = pair_to_merge[0] + pair_to_merge[1] # Proste łączenie stringów
    new_vocab_sequences = []
    for seq in vocab_sequences:
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair_to_merge:
                new_seq.append(new_token)
                i += 2  # Przeskocz o dwa, bo połączyliśmy parę
            else:
                new_seq.append(seq[i])
                i += 1
        new_vocab_sequences.append(new_seq)
    return new_vocab_sequences

def train_bpe(initial_sequences, num_merges):
    """
    Trenuje model BPE.
    'initial_sequences': lista list stringów (k-gramów).
    'num_merges': liczba operacji łączenia do wykonania.
    Zwraca:
        - bpe_vocab: finalny słownik tokenów BPE (string -> ID)
        - learned_merges: lista nauczonych operacji łączenia (par) w kolejności ich wykonania
        - tokenized_corpus_bpe_strings: korpus przetokenizowany przez BPE (listy stringów BPE)
    """
    print(f"Rozpoczęcie treningu BPE z {num_merges} operacjami łączenia...")
    current_sequences = [list(seq) for seq in initial_sequences] # Pracujemy na kopiach

    # Inicjalizacja słownika BPE: wszystkie unikalne początkowe k-gramy
    # oraz specjalne tokeny, które mogą być potrzebne później
    # Na tym etapie BPE_VOCAB będzie zawierał unikalne tokeny z `current_sequences`
    # plus potencjalne tokeny specjalne, jeśli zdecydujemy się je dodać do BPE.
    # Dla uproszczenia, BPE będzie działać tylko na dostarczonych k-gramach.
    
    base_tokens = set()
    for seq in current_sequences:
        for token in seq:
            base_tokens.add(token)
    
    # Słownik BPE będzie budowany dynamicznie podczas nauki łączeń
    # Na razie potrzebujemy listy nauczonych łączeń
    learned_merges = []

    for i in range(num_merges):
        stats = get_stats(current_sequences)
        if not stats:
            print("Brak par do dalszego łączenia.")
            break
        
        best_pair = max(stats, key=stats.get)
        current_sequences = merge_vocab(best_pair, current_sequences)
        learned_merges.append(best_pair)
        print(f"Iteracja {i+1}/{num_merges}: Połączono parę {best_pair} -> {best_pair[0]+best_pair[1]} (Częstość: {stats[best_pair]})")

    # Budowanie finalnego słownika BPE na podstawie tokenów występujących w przetworzonym korpusie
    final_bpe_tokens_set = set()
    for seq in current_sequences:
        for token in seq:
            final_bpe_tokens_set.add(token)
    
    # Dodajemy specjalne tokeny, które będą używane w wektoryzacji
    # Nawet jeśli BPE ich nie stworzyło, chcemy mieć dla nich ID
    BPE_SOS_TOKEN = "<BPE_SOS>"
    BPE_EOS_TOKEN = "<BPE_EOS>"
    BPE_PAD_TOKEN = "<BPE_PAD>" # Potrzebny do paddingu po tokenizacji BPE
    BPE_UNK_TOKEN = "<BPE_UNK>" # Dla tokenów BPE nieznalezionych w finalnym słowniku (nie powinno się zdarzyć)

    bpe_vocab_list = sorted(list(final_bpe_tokens_set)) # Sortowanie dla spójności ID
    
    bpe_vocab = {}
    bpe_id_counter = 0
    
    # Tokeny specjalne jako pierwsze
    for special_token in [BPE_PAD_TOKEN, BPE_UNK_TOKEN, BPE_SOS_TOKEN, BPE_EOS_TOKEN]:
        if special_token not in bpe_vocab: # Dodaj tylko jeśli nie zostały wygenerowane przez BPE
            bpe_vocab[special_token] = bpe_id_counter
            bpe_id_counter +=1
            
    for token in bpe_vocab_list:
        if token not in bpe_vocab:
            bpe_vocab[token] = bpe_id_counter
            bpe_id_counter += 1
            
    print(f"Trening BPE zakończony. Rozmiar słownika BPE: {len(bpe_vocab)} tokenów.")
    return bpe_vocab, learned_merges, current_sequences


def tokenize_sequence_with_bpe(kgram_sequence, learned_merges, bpe_vocab_final):
    """
    Tokenizuje pojedynczą sekwencję k-gramów (lista stringów)
    używając nauczonych operacji łączenia BPE.
    Zwraca listę stringów - tokenów BPE.
    """
    tokens = list(kgram_sequence) # Kopia
    
    for pair_to_merge in learned_merges:
        new_tokens = []
        i = 0
        merged_token_str = pair_to_merge[0] + pair_to_merge[1]
        # Sprawdzamy, czy połączony token istnieje w finalnym słowniku BPE.
        # To uproszczenie; pełne BPE stosuje łączenia iteracyjnie.
        # Tutaj zakładamy, że `learned_merges` są stosowane po kolei na oryginalnych k-gramach.
        # Bardziej poprawnie byłoby stosować je iteracyjnie na modyfikowanej liście tokenów.

        # Poniższa logika jest uproszczona i może nie odzwierciedlać w pełni
        # sposobu, w jaki BPE tokenizuje nowe sekwencje za pomocą *tylko* listy łączeń.
        # Bardziej typowe jest posiadanie słownika BPE i iteracyjne zastępowanie
        # najdłuższych pasujących tokenów BPE.
        # Dla tego przykładu, będziemy iteracyjnie przetwarzać listę tokenów.

        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair_to_merge:
                # Sprawdzamy, czy ten połączony token jest w ogóle częścią naszego wytrenowanego słownika BPE
                # (co oznacza, że to było "dobre" połączenie)
                if merged_token_str in bpe_vocab_final:
                    new_tokens.append(merged_token_str)
                    i += 2
                else: # Jeśli tego konkretnego połączenia nie ma w finalnym słowniku, nie łączymy
                    new_tokens.append(tokens[i])
                    i += 1
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens # Aktualizuj listę tokenów po każdej próbie scalenia

    return tokens


# --- Wektoryzacja ---

def one_hot_encode(bpe_token_ids_sequence, vocab_size):
    """
    Konwertuje sekwencję ID tokenów BPE na listę wektorów one-hot.
    """
    one_hot_vectors = []
    for token_id in bpe_token_ids_sequence:
        if token_id >= vocab_size: # Obsługa błędu, jeśli ID jest poza zakresem
            # Można tu rzucić błąd lub użyć wektora dla UNK
            vec = np.zeros(vocab_size, dtype=int)
            # Jeśli mamy ID dla UNK w bpe_vocab, można go tu użyć
            # np. vec[bpe_vocab[BPE_UNK_TOKEN]] = 1
            print(f"Ostrzeżenie: ID tokenu {token_id} poza zakresem słownika ({vocab_size}). Użyto wektora zerowego.")
        else:
            vec = np.zeros(vocab_size, dtype=int)
            vec[token_id] = 1
        one_hot_vectors.append(vec.tolist()) # Zapisujemy jako listy dla JSON
    return one_hot_vectors

def frequency_vector(bpe_token_ids_sequence, vocab_size):
    """
    Konwertuje sekwencję ID tokenów BPE na pojedynczy wektor częstości.
    """
    freq_vec = np.zeros(vocab_size, dtype=int)
    for token_id in bpe_token_ids_sequence:
        if token_id < vocab_size:
            freq_vec[token_id] += 1
        else:
            # Podobnie jak w one-hot, obsługa ID poza zakresem
             print(f"Ostrzeżenie: ID tokenu {token_id} poza zakresem słownika ({vocab_size}) przy tworzeniu wektora częstości.")
    return freq_vec.tolist() # Zapisujemy jako listę dla JSON


# --- Główna logika ---
def main_bpe_vectorizer():
    root = tk.Tk()
    root.withdraw()

    print("Wybierz plik JSON (output z k-gram tokenizera)...")
    input_json_path = filedialog.askopenfilename(
        title="Wybierz plik JSON z danymi k-gramów",
        filetypes=(("Pliki JSON", "*.json"), ("Wszystkie pliki", "*.*"))
    )
    if not input_json_path:
        print("Nie wybrano pliku. Zamykanie.")
        return

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            kgram_data = json.load(f)
        
        kgram_vocab = kgram_data['vocabulary']
        k_value = kgram_data['k_value']
        tokenized_kgram_id_sequences_padded = kgram_data['tokenized_sequences']
        
        print(f"Wczytano dane dla k={k_value} z {len(tokenized_kgram_id_sequences_padded)} sekwencjami.")
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku JSON: {e}")
        return

    # Odtworzenie sekwencji k-gramów (stringów) i usunięcie tokenów specjalnych/paddingu
    id_to_kgram_token = {v: k for k, v in kgram_vocab.items()}
    
    # Tokeny specjalne z pierwszego skryptu (musimy je znać, aby je usunąć)
    KG_PAD_TOKEN = "<PAD>"
    KG_SOS_TOKEN = "<SOS>"
    KG_EOS_TOKEN = "<EOS>"
    # KG_UNK_TOKEN nie usuwamy, bo jest to poprawny k-gram (lub jego reprezentacja)

    kgram_string_sequences_for_bpe = []
    for id_seq_padded in tokenized_kgram_id_sequences_padded:
        current_kgram_strings = []
        for token_id in id_seq_padded:
            token_str = id_to_kgram_token.get(token_id)
            if token_str and token_str not in [KG_PAD_TOKEN, KG_SOS_TOKEN, KG_EOS_TOKEN]:
                current_kgram_strings.append(token_str)
            elif token_str == KG_EOS_TOKEN: # Zakończ po napotkaniu EOS
                break 
        if current_kgram_strings: # Dodaj tylko jeśli sekwencja nie jest pusta po usunięciu specjalnych
            kgram_string_sequences_for_bpe.append(current_kgram_strings)
    
    if not kgram_string_sequences_for_bpe:
        print("Brak sekwencji k-gramów do przetworzenia po usunięciu tokenów specjalnych. Zamykanie.")
        return
    print(f"Przygotowano {len(kgram_string_sequences_for_bpe)} sekwencji k-gramów (stringów) dla BPE.")

    # Parametry BPE
    while True:
        try:
            num_merges_str = input("Podaj liczbę operacji łączenia dla BPE (np. 100): ").strip()
            num_merges = int(num_merges_str)
            if num_merges < 0:
                print("Liczba operacji łączenia musi być nieujemna.")
            else:
                break
        except ValueError:
            print("Nieprawidłowa wartość. Podaj liczbę całkowitą.")

    # Trening BPE
    # train_bpe zwraca: bpe_vocab, learned_merges, tokenized_corpus_bpe_strings
    # Gdzie tokenized_corpus_bpe_strings to korpus już przetworzony przez BPE (listy stringów)
    final_bpe_vocab, learned_bpe_merges, bpe_string_sequences_trained_corpus = train_bpe(kgram_string_sequences_for_bpe, num_merges)


    # Tokenizacja oryginalnych (niezmodyfikowanych przez trening BPE) sekwencji k-gramów
    # za pomocą nauczonych reguł BPE.
    # To jest ważne, jeśli chcemy przetworzyć nowe dane lub dane testowe w ten sam sposób.
    # W tym przypadku, `bpe_string_sequences_trained_corpus` to już wynik tokenizacji naszego korpusu treningowego.
    
    final_bpe_tokenized_sequences_as_strings = bpe_string_sequences_trained_corpus
    
    # Dodanie tokenów SOS/EOS do sekwencji BPE i konwersja na ID
    BPE_SOS_TOKEN = "<BPE_SOS>" # Upewnij się, że te stałe są zdefiniowane
    BPE_EOS_TOKEN = "<BPE_EOS>"
    BPE_PAD_TOKEN = "<BPE_PAD>"
    BPE_UNK_TOKEN = "<BPE_UNK>"


    bpe_sos_id = final_bpe_vocab[BPE_SOS_TOKEN]
    bpe_eos_id = final_bpe_vocab[BPE_EOS_TOKEN]
    bpe_unk_id = final_bpe_vocab[BPE_UNK_TOKEN]

    bpe_token_id_sequences = []
    for bpe_str_seq in final_bpe_tokenized_sequences_as_strings:
        ids = [bpe_sos_id]
        for bpe_token_str in bpe_str_seq:
            ids.append(final_bpe_vocab.get(bpe_token_str, bpe_unk_id))
        ids.append(bpe_eos_id)
        bpe_token_id_sequences.append(ids)

    # Padding sekwencji BPE (ID)
    max_len_bpe = 0
    if bpe_token_id_sequences:
        max_len_bpe = max(len(s) for s in bpe_token_id_sequences)
    
    bpe_pad_id = final_bpe_vocab[BPE_PAD_TOKEN]
    padded_bpe_id_sequences = []
    for id_seq in bpe_token_id_sequences:
        padding_needed = max_len_bpe - len(id_seq)
        padded_bpe_id_sequences.append(id_seq + ([bpe_pad_id] * padding_needed))
    
    print(f"Tokenizacja BPE i padding zakończone. Maksymalna długość sekwencji BPE: {max_len_bpe}")

    # Wybór metody wektoryzacji
    while True:
        vectorization_choice = input("Wybierz metodę wektoryzacji:\n  1) One-hot encoding\n  2) Wektor częstościowy (Frequency vector)\nWybór (1 lub 2): ").strip()
        if vectorization_choice in ["1", "2"]:
            break
        else:
            print("Nieprawidłowy wybór. Wpisz 1 lub 2.")

    vectorized_sequences = []
    bpe_vocab_size = len(final_bpe_vocab)

    if vectorization_choice == "1":
        print("Wybrano: One-hot encoding.")
        # Dla one-hot, każda sekwencja (lista ID BPE) staje się listą wektorów one-hot
        for bpe_id_seq_padded in padded_bpe_id_sequences:
            # Nie wektoryzujemy tokenów PAD dla one-hot, chyba że jest to wymagane
            # Zazwyczaj model sam uczy się ignorować padding lub używa maskowania
            # Dla celów demonstracyjnych, przetworzymy całą dopełnioną sekwencję
            one_hot_representation = one_hot_encode(bpe_id_seq_padded, bpe_vocab_size)
            vectorized_sequences.append(one_hot_representation)
        vectorization_method_name = "one_hot"
    else: # vectorization_choice == "2"
        print("Wybrano: Wektor częstościowy.")
        # Dla wektora częstości, każda sekwencja (lista ID BPE) staje się jednym wektorem
        for bpe_id_seq_padded in padded_bpe_id_sequences:
            # Usuwamy padding PRZED obliczeniem wektora częstości,
            # ponieważ tokeny PAD nie powinny wpływać na częstości rzeczywistych tokenów.
            # Można też zdecydować się na uwzględnienie częstości SOS/EOS.
            # Tutaj usuniemy PAD, SOS, EOS dla "czystszego" wektora częstości samych danych.
            
            ids_for_frequency = [
                token_id for token_id in bpe_id_seq_padded 
                if token_id not in [bpe_pad_id, bpe_sos_id, bpe_eos_id]
            ]
            if not ids_for_frequency: # Jeśli po usunięciu została pusta sekwencja
                vectorized_sequences.append(np.zeros(bpe_vocab_size, dtype=int).tolist())
            else:
                vectorized_sequences.append(frequency_vector(ids_for_frequency, bpe_vocab_size))
        vectorization_method_name = "frequency_vector"
        
    print("Wektoryzacja zakończona.")

    # Zapis wyników
    output_filename_suggestion = os.path.splitext(os.path.basename(input_json_path))[0] + f"_bpe_{vectorization_method_name}.json"
    
    output_save_path = filedialog.asksaveasfilename(
        title="Zapisz zwektoryzowane dane BPE",
        initialfile=output_filename_suggestion,
        defaultextension=".json",
        filetypes=(("Pliki JSON", "*.json"), ("Wszystkie pliki", "*.*"))
    )

    if not output_save_path:
        print("Nie wybrano lokalizacji zapisu. Anulowano.")
        return

    output_data_bpe = {
        "bpe_k_value_original": k_value,
        "bpe_num_merges": num_merges,
        "bpe_vocabulary": final_bpe_vocab,
        "bpe_learned_merges_in_order": learned_bpe_merges, # Lista par [(t1, t2), (t3,t4)...]
        "vectorization_method": vectorization_method_name,
        # "original_kgram_sequences_for_bpe_input": kgram_string_sequences_for_bpe, # Opcjonalnie, dla debugowania
        # "bpe_tokenized_sequences_strings": final_bpe_tokenized_sequences_as_strings, # Opcjonalnie
        "padded_bpe_token_id_sequences": padded_bpe_id_sequences, # Dla referencji
        "vectorized_sequences": vectorized_sequences
    }

    try:
        with open(output_save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data_bpe, f, indent=4)
        print(f"Wyniki BPE i wektoryzacji zapisano do: {output_save_path}")
    except Exception as e:
        print(f"Błąd podczas zapisu pliku wynikowego: {e}")

if __name__ == "__main__":
    main_bpe_vectorizer()
