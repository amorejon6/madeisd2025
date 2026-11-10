import polars as pl
import pandas as pd
import numpy as np
import warnings
import os
import csv
import shutil
import hashlib
import concurrent.futures
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from usearch.index import Index

warnings.filterwarnings("ignore")

TIMEOUT = 120 # Tiempo límite para cada archivo (en segundos)

model = SentenceTransformer('jinaai/jina-embeddings-v2-base-es', trust_remote_code=True, device='cuda')

def process_dataframe(df):
    df_pandas = df.to_pandas()
    rows = np.array([
        model.encode(" ".join(map(str, filter(pd.notna, row))))
        for row in df_pandas.values.tolist()
    ])
    return rows

def generate_numeric_id(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    return str(hash_int)[:16]

def save_in_index(df, file, rows, index_c, index_h):
    headers = " ".join(df.columns.tolist())
    emb_headers = model.encode([headers])

    emb_content = np.mean(rows, axis=0)

    dataframe_id = generate_numeric_id(file)
    index_c.add(int(dataframe_id), emb_content)
    index_h.add(int(dataframe_id), emb_headers)

def clustering(df, numerical_vectors):
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(numerical_vectors)
    centroids = kmeans.cluster_centers_
    distances = cdist(numerical_vectors, centroids, 'euclidean')
    positions = np.argmin(distances, axis=0)
    final_rows = df[positions]
    return final_rows

def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename, encoding="ISO-8859-1") as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter

def process_file(file, directory, directory_save):
    sep = find_delimiter(directory + file)
    df = pl.read_csv(directory + file, separator=sep, encoding="utf-8", ignore_errors=True, n_rows=5000)

    vectors = process_dataframe(df)
    # print(vectors)
    final_df = clustering(df, vectors)
    final_df_pandas = final_df.to_pandas()

    os.makedirs(directory_save + 'polars/', exist_ok=True)
    final_df_pandas.to_csv(directory_save + 'polars/' + file.split('.')[0] + '_reduced.csv', sep=sep, index=False)

    final_df = df.sample(n=10, seed=42)
    final_df_pandas = final_df.to_pandas()
    os.makedirs(directory_save + 'random/', exist_ok=True)
    final_df_pandas.to_csv(directory_save + 'random/' + file.split('.')[0] + '_reduced.csv', sep=sep, index=False)

    final_df = df.head(10)
    final_df_pandas = final_df.to_pandas()
    os.makedirs(directory_save + 'ten/', exist_ok=True)
    final_df_pandas.to_csv(directory_save + 'ten/' + file.split('.')[0] + '_reduced.csv', sep=sep, index=False)

def main():
    directory = 'dataset/'
    files = os.listdir(directory)
    directory_save = root + 'reduced_tables_embeddings/normal_experiment/'

    for file in tqdm(files):
        if not os.path.exists(directory_save + 'ten/' + file.split('.')[0] + '_reduced.csv'):
            print(file)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(process_file, file, directory, directory_save)
                    future.result(timeout=TIMEOUT)  # Limitar a TIMEOUT segundos
            except concurrent.futures.TimeoutError:
                print(f"El archivo {file} excedió el tiempo límite de {TIMEOUT} segundos.")
            except Exception as e:
                print(f'Error con el archivo {file}.\n', e)
        else:
            print(f"File '{file}' has already been reduced")

if __name__ == "__main__":
    main()