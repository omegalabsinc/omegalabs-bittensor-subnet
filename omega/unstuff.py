import torch
from transformers import pipeline
from typing import Tuple
import random
import torch.nn.functional as F
from omega.imagebind_wrapper import (
    split_text_by_token_limit,
    SimpleTokenizer,
    BPE_PATH,
    split_text_by_token_limit,
)
CHUNK_SIZE = 60
TOKENIZER = SimpleTokenizer(bpe_path=BPE_PATH, context_length=10000)

UNSTUFF = pipeline("text-classification", "jondurbin/unstuffer-v0.2", device="cuda" if torch.cuda.is_available() else "cpu")

def is_stuffed(description: str) -> Tuple[bool, float]:
    result = UNSTUFF(description, truncation=True, max_length=512)
    stuffed = False if int(result[0]["label"]) == 1 else True
    confidence = result[0]["score"]
    if stuffed and confidence > 0.75:
        print(f"Detected stuffed description [{confidence=}]: {description}")
    elif not stuffed and random.random() <= 0.01:
        print(f"Description does not appear to be stuffed [{confidence=}]: {description}")
    return stuffed, confidence

def check_extraneous_chunks(description, video_emb, audio_emb, imagebind):
    text_chunks = [
        chunk
        for chunk in split_text_by_token_limit(description, TOKENIZER, CHUNK_SIZE)
        if len(TOKENIZER(chunk)) >= 5
    ]
    if len(text_chunks) <= 1:
        return 0.0
    similarities = []
    for text in text_chunks:
        text_emb = imagebind.embed_text([text]).to("cpu")
        v_cosim = F.cosine_similarity(
            torch.tensor(video_emb), text_emb
        ).tolist()[0]
        a_cosim = F.cosine_similarity(
            torch.tensor(audio_emb), text_emb
        ).tolist()[0]
        similarities.append((v_cosim + a_cosim) / 2)
    best = max(similarities)
    low_quality = 0
    really_bad = 0
    for idx in range(len(similarities)):
        similarity = similarities[idx]
        text = text_chunks[idx]
        if similarity < best * 0.6:
            low_quality += 1
        if similarity < 0.12:
            really_bad += 1
    return really_bad, low_quality, len(similarities)
