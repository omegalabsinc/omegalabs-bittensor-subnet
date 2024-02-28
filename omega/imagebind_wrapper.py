from typing import List, BinaryIO

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.models.multimodal_preprocessors import SimpleTokenizer
from pydantic import BaseModel
import torch

from omega import video_utils


BPE_PATH = "./omega/bpe/bpe_simple_vocab_16e6.txt.gz"


class Embeddings(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    video: torch.Tensor
    audio: torch.Tensor
    description: torch.Tensor


def load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


class ImageBind:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.imagebind = imagebind_model.imagebind_huge(pretrained=True)
        self.imagebind.eval()
        self.imagebind.to(self.device)

    @torch.no_grad()
    def embed(self, descriptions: List[str], video_files: List[BinaryIO]) -> Embeddings:
        audio_files = [video_utils.copy_audio(video_file.name) for video_file in video_files]
        audio_filepaths = [audio_file.name for audio_file in audio_files]
        video_filepaths = [video_file.name for video_file in video_files]
        try:
            video_data = data.load_and_transform_video_data(video_filepaths, self.device)
            audio_data = data.load_and_transform_audio_data(audio_filepaths, self.device)
            inputs = {
                ModalityType.TEXT: load_and_transform_text(descriptions, self.device),
                ModalityType.VISION: video_data,
                ModalityType.AUDIO: audio_data,
            }
            embeddings = self.imagebind(inputs)
            return Embeddings(
                video=embeddings[ModalityType.VISION],
                audio=embeddings[ModalityType.AUDIO],
                description=embeddings[ModalityType.TEXT]
            )
        finally:
            for audio_file in audio_files:
                audio_file.close()

    @torch.no_grad()
    def embed_text(self, texts: List[str]) -> torch.Tensor:
        return self.imagebind({
            ModalityType.TEXT: data.load_and_transform_text(texts, self.device),
        })[ModalityType.TEXT]
