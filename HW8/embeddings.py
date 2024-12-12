from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import chromadb
import csv
import numpy as np

if __name__ == "__main__":
    dataset = load_dataset("hazyresearch/LoCoV1-Documents")
    model = AutoModelForSequenceClassification.from_pretrained(
    "togethercomputer/m2-bert-80M-2k-retrieval",
    trust_remote_code=True
    )
    max_seq_length = 1024
    tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    model_max_length=max_seq_length
    )
    articles = dataset['test']['passage']
    num_samples = len(articles)
    with open("embeddings.csv","w") as f:
        breakpoint()
        for i in range(num_samples):
            input_ids = tokenizer(dataset['test']['passage'][i],  return_tensors="pt",  padding="max_length",  return_token_type_ids=False,  truncation=True,  max_length=max_seq_length)
            encoding = model(input_ids['input_ids'])
            encoding_np = encoding['sentence_embedding'].detach().numpy()
            np.savetxt(f, encoding_np)
    breakpoint()
