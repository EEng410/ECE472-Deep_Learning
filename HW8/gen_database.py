from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import chromadb
import csv
import numpy as np
import pandas as pd

dataset = load_dataset("hazyresearch/LoCoV1-Documents")
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.Client()

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.create_collection("DocumentEmbeddings")

# I generate my stuff locally, I will store it locally though
local_client = chromadb.PersistentClient(path = "/content")

copied_collection = client.get_or_create_collection("DocumentEmbeddings")

# load embeddings
embeddings_df = pd.read_csv('embeddings.csv', header = None, delimiter = ',')
embeddings_np = embeddings_df[0].to_numpy()

for i in range(len(embeddings_df[0])):
  collection.add(
      embeddings = np.fromstring(embeddings_np[i], dtype=float, sep=' '),
      documents = dataset['test']['passage'][i],
      ids = dataset['test']['pid'][i]
  )
  
model = AutoModelForSequenceClassification.from_pretrained(
"togethercomputer/m2-bert-80M-2k-retrieval",
trust_remote_code=True
)
max_seq_length = 1024
tokenizer = AutoTokenizer.from_pretrained(
"bert-base-uncased",
model_max_length=max_seq_length
)

query = "What is machine learning?"
input_ids = tokenizer(query,  return_tensors="pt",  padding="max_length",  return_token_type_ids=False,  truncation=True,  max_length=max_seq_length)
encoding = model(input_ids['input_ids'])
encoding_np = encoding['sentence_embedding'].detach().numpy()
results = collection.query(
    query_embeddings = encoding_np,
    n_results = 10
)

print(results['documents'])