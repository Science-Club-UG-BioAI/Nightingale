# Nightingale

Jak używać nowego skryptu:
Uruchom pierwszy skrypt (k-gram tokenizer):
Podaj mu plik CSV z sekwencjami białek.
Wybierz wartość k.
Zapisz wynik w formacie JSON. Zapamiętaj, gdzie zapisałeś ten plik.
Uruchom drugi skrypt (BPE i wektoryzacja):
Skrypt poprosi Cię o wskazanie pliku JSON wygenerowanego w poprzednim kroku.
Następnie zapyta o liczbę operacji łączenia (merges) dla algorytmu BPE (np. 50, 100, 500 – im więcej, tym potencjalnie dłuższe i bardziej złożone tokeny BPE powstaną, ale też dłużej potrwa trening).
Po treningu BPE, skrypt zapyta o wybór metody wektoryzacji:
1 dla One-hot encoding.
2 dla Wektora częstościowego.
Na koniec poprosi o wskazanie miejsca zapisu finalnego pliku JSON, który będzie zawierał:
Informacje o oryginalnym k i liczbie łączeń BPE.
Nauczony słownik BPE (mapowanie tokenów BPE na ID).
Listę nauczonych operacji łączenia.
Zastosowaną metodę wektoryzacji.
Przetokenizowane (przez BPE i dopełnione) sekwencje jako listy ID.
Finalne zwektoryzowane sekwencje.

