import requests
import pandas as pd
import numpy as np


def clear_text(text):
    if not isinstance(text, str):
        return text
    text = text.translate(str.maketrans(':|/\\;"', ' '*6))
    return ' '.join(word for word in text.split() if not (word.startswith(('(PubMed', 'PubMed'))))


def get_uniprot_data():
    link = None
    results = []
    url = 'https://rest.uniprot.org/uniprotkb/search'
    params = {
            'query': 'organism_id:562 AND length:[1 TO 800]',
            'format': 'json',
            'size': 500,
        }
    while True:
        response = requests.get(link) if link else requests.get(url, params=params)
        if response.status_code != 200:
            print("Błąd:", response.status_code)
            break

        data = response.json()
        results_batch = 0

        for entry in data.get('results', []):
            id_uniprot = entry.get('primaryAccession', np.nan)
            sequence = entry.get('sequence', {}).get('value', np.nan)
            # protein_description = entry.get('proteinDescription').get('value', np.nan)
        
            func = next(
                (clear_text(comment['texts'][0]['value']) for comment in entry.get('comments', [])
                if comment.get('commentType') == 'FUNCTION' and comment.get('texts') and comment['texts'][0].get('value')), 
                np.nan
            )
            results.append([id_uniprot, sequence, func])
            results_batch += 1

        print(f"Pobrano {results_batch} rekordów, razem: {len(results)}")

        link = response.headers.get('Link', '')
        if 'rel="next"' in link:
            link = link[link.find('<')+1 : link.find('>')]
        else:
            link = ''
            break


    df = pd.DataFrame(results, columns=['uniprot ID', 'sequence', 'function'])
    df = df.dropna(how='any')
    df.to_csv('ecoli.csv', index=False, sep='|')
    df_seq = df.loc[:,'sequence']
    df_seq.to_csv('ecoli_seq.csv', index=False, sep='|')
    print(f'Ilość wejść z funkcją: {len(df)}')
    


get_uniprot_data()