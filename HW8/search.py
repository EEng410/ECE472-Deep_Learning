from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import chromadb
import csv
import numpy as np
import pandas as pd
from keybert import KeyBERT

if __name__ == '__main__':
    # Just in case the directories look strange, I originally did this through 
    # Colab and I had some trouble downloading my chroma database for some reason
    # In general this works just fine though
    client = chromadb.PersistentClient(path = "content/ChromaNew")

    collection = client.get_or_create_collection("please")

    model = AutoModelForSequenceClassification.from_pretrained(
    "togethercomputer/m2-bert-80M-2k-retrieval",
    trust_remote_code=True
    )
    max_seq_length = 1024
    tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    model_max_length=max_seq_length
    )
    query = input('Search: ')
    input_ids = tokenizer(query,  return_tensors="pt",  padding="max_length",  return_token_type_ids=False,  truncation=True,  max_length=max_seq_length)
    encoding = model(input_ids['input_ids'])
    encoding_np = encoding['sentence_embedding'].detach().numpy()
    results = collection.query(
        query_embeddings = encoding_np,
        n_results = 50
    )
    
    # Do a keyword search on the results and find which ones are relevant to keywords in the query
    kw_model = KeyBERT()
    query_keywords = kw_model.extract_keywords(query)
    
    keywords_str = [query_keywords[i][0] for i in range(len(query_keywords))]
    
    improved_results = []
    
    num_appearances = np.zeros(shape = (len(results['documents'][0]), 1))
    # Iterate through passages and only show those that use the relevant keywords
    for i in range(len(results['documents'][0])):
        for j in range(len(keywords_str)):
            # Do a case insensitive search
            num_appearances[i] += results['documents'][0][i].lower().count(keywords_str[j])
        if num_appearances[i] >= 1:
            improved_results.append(results['documents'][0][i])
    # Sort according to keyword frequency
    improved_results = np.array(improved_results)[np.flip(np.argsort(num_appearances[num_appearances >= 1]))].tolist()
    with open("output.txt", "w") as text_file:
        text_file.write(query)
        for i in range(len(improved_results)):
            num = 1+i
            text_file.write(f"\n \n \n Result {num} \n \n \n")
            text_file.write(improved_results[i][0:2499])
    breakpoint()