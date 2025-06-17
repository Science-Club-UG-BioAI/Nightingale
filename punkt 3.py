#zaimportowano bibliotekę pandas
import pandas as pd

#wczytano plik CSV z danymi o białkach E. coli
df = pd.read_csv('ecoli.csv', sep='|')

#stworzono słownik, gdzie kluczem jest uniprot ID, a wartością sekwencja białka
slownik = {}

#otwarto plik z danymi o rodzinach domen Pfam
with open('pfam_rodzina.txt', 'r') as plik:
    for linia in plik:
        if not linia.startswith('#'): #pomijanie linii komentarzy
            linia = linia.strip() #usunięto białe znaki z początku i końca linii
            czesci = linia.split() #podzielono linię na części

            if len(czesci) >= 4:
                query_id = czesci[3]
                pfam_id = czesci[1]
                domain_name = czesci[0]

                if query_id not in slownik:
                    slownik[query_id] = []

                slownik[query_id].append((pfam_id, domain_name))

print("Słownik pfam_id -> (pfam_id, domain_name):")
for k, v in slownik.items():
    print(f"{k}: {v}")

df['Domena'] = df['uniprot ID'].map(lambda x: slownik.get(x, []))
df.to_csv('ecoli2.csv', index=False, sep='|')
print("Dane zostały zapisane do pliku 'ecoli2.csv'.")


