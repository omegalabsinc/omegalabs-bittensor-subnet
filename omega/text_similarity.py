import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model_path = "Alibaba-NLP/gte-large-en-v1.5"
revision = "104333d6af6f97649377c2afbde10a7704870c7b"
TOKENIZER = AutoTokenizer.from_pretrained(model_path, revision=revision)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = AutoModel.from_pretrained(model_path, trust_remote_code=True, revision=revision).to(DEVICE)
MODEL.eval()

def get_text_similarity_score(text_0, text_1):
    tokens = TOKENIZER([text_0, text_1], max_length=1024, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
    outputs = MODEL(**tokens)
    embeddings = outputs.last_hidden_state[:, 0]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T) * 100
    return min(1.0, (scores.tolist()[0][0] / 100) ** 2)
