from transformers import BertTokenizer, BertModel
from pathlib import Path

model_path = Path("bert-base-uncased").resolve()
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

print(tokenizer.tokenize("I love machine learning."))