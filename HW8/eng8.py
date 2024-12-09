from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import chromadb

if __name__ == "__main__":
    dataset = load_dataset("hazyresearch/LoCoV1-Documents")
    model = AutoModelForSequenceClassification.from_pretrained(
    "togethercomputer/m2-bert-80M-2k-retrieval",
    trust_remote_code=True
    )
    max_seq_length = 2048
    tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    model_max_length=max_seq_length
    )
