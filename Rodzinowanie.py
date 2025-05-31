import pandas as pd


df = pd.read_csv("C:\\Studia\\Progranmy\\kodON\\Słowik\\ecoli.csv", sep="|")
pfam_map = {}

with open("C:\\Studia\\Progranmy\\kodON\\PfamFiles\\pfam_rodzina.txt") as f:
    for line in f:
        if not line.startswith("#"):
            parts = line.strip().split()
            query_id = parts[0]
            pfam_id = parts[1]
            domain = parts[2]
            pfam_map.setdefault(query_id, []).append((pfam_id, domain))

df["Pfam_hits"] = df["uniprot ID"].map(lambda x: pfam_map.get(x, []))
df.to_csv("C:\\Studia\\Progranmy\\kodON\\Słowik\\proteins_with_pfam.csv", index=False)
