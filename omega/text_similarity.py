import math
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model_path = "Alibaba-NLP/gte-large-en-v1.5"
revision = "104333d6af6f97649377c2afbde10a7704870c7b"
TOKENIZER = AutoTokenizer.from_pretrained(model_path, revision=revision)
MODEL = AutoModel.from_pretrained(model_path, trust_remote_code=True, revision=revision)

def similarity(text_0, text_1):
    tokens = TOKENIZER([text_0, text_1], max_length=1024, padding=True, truncation=True, return_tensors='pt')
    outputs = MODEL(**tokens)
    embeddings = outputs.last_hidden_state[:, 0]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T) * 100
    return min(1.0, math.sqrt(scores.tolist()[0][0] / 100))
