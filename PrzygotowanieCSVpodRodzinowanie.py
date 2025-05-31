import pandas as pd

df = pd.read_csv("C:\\Studia\\Progranmy\\kodON\\Słowik\\ecoli.csv", sep="|")

with open("C:\\Studia\\Progranmy\\kodON\\Słowik\\sequencesHMMER.fasta", "w") as f:
    for _, row in df.iterrows():
        f.write(f">{row['uniprot ID']}\n{row['sequence']}\n")










