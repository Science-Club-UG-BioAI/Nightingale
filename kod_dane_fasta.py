import pandas as pd

df = pd.read_csv("/Users/Emilka/PycharmProjects/pythonProject/Nightingale/ecoli.csv", sep="|")
df = df[['uniprot ID', 'sequence']]
df['sequence'] = df['sequence'].str.replace(r'\s+', '', regex=True)

with open("staramsie.fasta", "w") as fasta_file:
    for _, row in df.iterrows():
        fasta_file.write(f">{row['uniprot ID']}\n")
        fasta_file.write(f"{row['sequence']}\n")

print("Dane powinny być w pliku staramsie.fasta")
