import os
import asyncio
import functools
from typing import List, BinaryIO, Optional

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.models.multimodal_preprocessors import SimpleTokenizer, TextPreprocessor
from pydantic import BaseModel
import torch

from omega import video_utils


BPE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe", "bpe_simple_vocab_16e6.txt.gz")
V2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".checkpoints", "videobind.pth")
TOKENIZER = SimpleTokenizer(bpe_path=BPE_PATH)
TOKENIZER_V2 = SimpleTokenizer(bpe_path=BPE_PATH, context_length=1024)

class Embeddings(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    video: Optional[torch.Tensor]
    audio: Optional[torch.Tensor]
    description: Optional[torch.Tensor]


def load_and_transform_text(text, device, v2=True):
    if text is None:
        return None
    tokenizer = TOKENIZER_V2 if v2 else TOKENIZER
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def run_async(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


class ImageBind:
    def __init__(self, v2=True):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.v2 = v2
        if self.v2:
            if not os.path.exists(V2_PATH):
                os.makedirs(os.path.dirname(V2_PATH), exist_ok=True)
                torch.hub.download_url_to_file(
                    "https://huggingface.co/jondurbin/videobind-v0.1/resolve/main/videobind.pth",
                    V2_PATH,
                    progress=True,
                )
            self.imagebind = torch.load(V2_PATH)
        else:
            self.imagebind = imagebind_model.imagebind_huge(pretrained=True)
        self.imagebind.eval()
        self.imagebind.to(self.device)

    def get_inputs(self, description: str, video_file: BinaryIO) -> dict:
        audio_file = video_utils.copy_audio(video_file.name)
        try:
            duration = video_utils.get_video_duration(video_file.name)
            video_data = data.load_and_transform_video_data(
                [video_file.name],
                self.device,
            )
            audio_data = data.load_and_transform_audio_data(
                [audio_file.name],
                self.device,
            )
            inputs = {
                ModalityType.TEXT: load_and_transform_text([description], self.device, v2=self.v2),
                ModalityType.VISION: video_data,
                ModalityType.AUDIO: audio_data,
            }
            return inputs
        finally:
            audio_file.close()

    @torch.no_grad()
    def embed(self, descriptions: List[str], video_files: List[BinaryIO]) -> Embeddings:
        return_value = None
        for idx in range(len(descriptions)):
            inputs = self.get_inputs(descriptions[idx], video_files[idx])
            embeddings = self.imagebind(inputs)
            if not return_value:
                return_value = Embeddings(
                    video=embeddings[ModalityType.VISION],
                    audio=embeddings[ModalityType.AUDIO],
                    description=embeddings[ModalityType.TEXT]
                )
            else:
                return_value.video = torch.cat((return_value.video, embeddings[ModalityType.VISION]))
                return_value.audio = torch.cat((return_value.audio, embeddings[ModalityType.AUDIO]))
                return_value.description = torch.cat((return_value.description, embeddings[ModalityType.TEXT]))
        return return_value

    @torch.no_grad()
    def embed_only_video(self, video_files: List[BinaryIO]) -> Embeddings:
        video_filepaths = [video_file.name for video_file in video_files]
        durations = [video_utils.get_video_duration(f.name) for f in video_files]
        embeddings = self.imagebind({
            ModalityType.VISION: [
                data.load_and_transform_video_data(
                    [video_filepaths[idx]],
                    self.device,
                )[0]
                for idx in range(len(video_filepaths))
            ]
        })
        return Embeddings(
            video=embeddings[ModalityType.VISION],
        )
        
    @torch.no_grad()
    def embed_video_and_text(self, video_files: List[BinaryIO], descriptions: List[str]) -> Embeddings:
        video_filepaths = [video_file.name for video_file in video_files]
        durations = [video_utils.get_video_duration(f.name) for f in video_files]
        embeddings = self.imagebind({
            ModalityType.VISION: [
                data.load_and_transform_video_data(
                    [video_filepaths[idx]],
                    self.device,
                )[0]
                for idx in range(len(video_filepaths))
            ],
            ModalityType.TEXT: load_and_transform_text(descriptions, self.device, v2=self.v2)
        })
        return Embeddings(
            video=embeddings[ModalityType.VISION],
            description=embeddings[ModalityType.TEXT]
        )
   
    @torch.no_grad()
    def embed_text(self, texts: List[str]) -> torch.Tensor:
        return self.imagebind({
            ModalityType.TEXT: load_and_transform_text(texts, self.device, v2=self.v2),
        })[ModalityType.TEXT]

    @torch.no_grad()
    async def embed_async(self, descriptions: List[str], video_files: List[BinaryIO]) -> Embeddings:
        inputs = self.get_inputs(descriptions, video_files)  # cannot be async
        embeddings = await run_async(self.imagebind, inputs)
        return Embeddings(
            video=embeddings[ModalityType.VISION],
            audio=embeddings[ModalityType.AUDIO],
            description=embeddings[ModalityType.TEXT]
        )

    async def embed_text_async(self, texts: List[str]) -> torch.Tensor:
        return await run_async(self.embed_text, texts)
