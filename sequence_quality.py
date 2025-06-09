import pandas as pd
from keywords import excluding_keywords, including_keywords

df = pd.read_csv('https://raw.githubusercontent.com/Science-Club-UG-BioAI/Nightingale/refs/heads/DATA/ecoli.csv', sep='|')
print(df.head())

sequences_functions = df['function'].tolist()
print(sequences_functions[:1])

# checking the quality of sequences based on given keywords, throwing out sequences that do not match the criteria
def check_sequence_quality(sequences_functions):
  filtered_sequences = []
  for seq in sequences_functions:
    if any(keyword in seq for keyword in excluding_keywords):
      sequences_functions.remove(seq)
      continue
    elif not any(keyword in seq for keyword in including_keywords):
      sequences_functions.remove(seq)
      continue
    filtered_sequences.append(seq)
  return filtered_sequences


# running program
if __name__ == "__main__":
  filtered_sequences = check_sequence_quality(sequences_functions)
  print(f"Filtered sequences: {filtered_sequences}")
  print(f"Total filtered sequences: {len(filtered_sequences)}")
  
  # saving new sequences to a new csv file
  good_quality_sequences = pd.DataFrame(filtered_sequences, columns=['function'])
  good_quality_sequences.to_csv('filtered_ecoli.csv', index=False, sep='|')


# ouput for given criteria:
######## = Total filtered sequences: 9010