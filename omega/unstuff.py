import torch
from transformers import pipeline
from typing import Tuple

UNSTUFF = pipeline("text-classification", "jondurbin/unstuffer-v0.2", device="cuda" if torch.cuda.is_available() else "cpu")

def is_stuffed(description: str) -> Tuple[bool, float]:
    result = UNSTUFF(description)
    stuffed = False if int(result[0]["label"]) == 1 else True
    confidence = result[0]["score"]
    if stuffed:
        print(f"Detected stuffed description [{confidence=}]: {description}")
    else:
        print(f"Description does not appear to be stuffed [{confidence=}]: {description}")
    return stuffed, confidence
