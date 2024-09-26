import numpy as np
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
LENGTH_TOKENIZER = SimpleTokenizer(bpe_path=BPE_PATH, context_length=1024)
TOKEN_CHUNK_SIZE = 74

class Embeddings(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    video: Optional[torch.Tensor]
    audio: Optional[torch.Tensor]
    description: Optional[torch.Tensor]


def load_and_transform_text(text, device):
    if text is None:
        return None
    tokens = [TOKENIZER(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def split_text_by_token_limit(text, tokenizer, max_tokens=TOKEN_CHUNK_SIZE):
    def fits_in_token_limit(text_segment):
        tokens = tokenizer(text_segment)
        tokens = tokens[tokens != 0][1:-1].tolist()
        return len(tokens) <= max_tokens

    def recursive_split(text, delimiters):
        if fits_in_token_limit(text):
            return [text]
        if not delimiters:
            return split_by_tokens(text)
        delimiter = delimiters[0]
        parts = text.split(delimiter)
        result = []
        current_segment = ""
        for part in parts:
            candidate_segment = current_segment + (delimiter if current_segment else '') + part
            if fits_in_token_limit(candidate_segment):
                current_segment = candidate_segment
            else:
                if current_segment:
                    result.append(current_segment)
                current_segment = part
        if current_segment:
            result.append(current_segment)
        final_result = []
        for segment in result:
            if fits_in_token_limit(segment):
                final_result.append(segment)
            else:
                final_result.extend(recursive_split(segment, delimiters[1:]))
        return final_result

    def split_by_tokens(text):
        tokens = tokenizer(text)
        tokens = tokens[tokens != 0][1:-1]
        segments = []
        for i in range(0, len(tokens), max_tokens):
            segment_tokens = tokens[i:i + max_tokens]
            segment_text = tokenizer.decode(segment_tokens)
            segments.append(segment_text)
        return segments

    return recursive_split(text, ['\n', '.', '!', '?', ',', ' '])

def load_and_transform_text_chunks(text, device):
    if not text:
        return []
    all_tokens = LENGTH_TOKENIZER(text)
    all_tokens = all_tokens[all_tokens != 0][1:-1].tolist()

    return [
        load_and_transform_text([segment], device)
        for segment in split_text_by_token_limit(text, LENGTH_TOKENIZER)
    ]

def run_async(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


class ImageBind:
    def __init__(self, device="cuda:0", v2=False):
        self.device = device
        self.v2 = v2
        if v2:
            if not os.path.exists(V2_PATH):
                os.makedirs(os.path.dirname(V2_PATH), exist_ok=True)
                torch.hub.download_url_to_file(
                    "https://huggingface.co/jondurbin/videobind-v0.2/resolve/main/videobind.pth",
                    V2_PATH,
                    progress=True,
                )
            self.imagebind = torch.load(V2_PATH)
        else:
            self.imagebind = imagebind_model.imagebind_huge(pretrained=True)
        self.imagebind.eval()
        self.imagebind.to(self.device)

    def generate_text_embeddings(self, text: str):
        if not self.v2:
            return self.imagebind({
                ModalityType.TEXT: load_and_transform_text([text], self.device)
            })[ModalityType.TEXT]
        chunks = load_and_transform_text_chunks(text, self.device)
        embeddings = [
            self.imagebind({ModalityType.TEXT: chunk})[ModalityType.TEXT]
            for chunk in chunks
        ]
        return torch.mean(torch.stack(embeddings), dim=0)

    def get_inputs(self, video_file: BinaryIO) -> dict:
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
            inputs = self.get_inputs(video_files[idx])
            embeddings = self.imagebind(inputs)
            text_embeddings = self.generate_text_embeddings(descriptions[idx])
            if not return_value:
                return_value = Embeddings(
                    video=embeddings[ModalityType.VISION],
                    audio=embeddings[ModalityType.AUDIO],
                    description=text_embeddings,
                )
            else:
                return_value.video = torch.cat((return_value.video, embeddings[ModalityType.VISION]))
                return_value.audio = torch.cat((return_value.audio, embeddings[ModalityType.AUDIO]))
                return_value.description = torch.cat((return_value.description, text_embeddings))
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
        })
        description_embeddings = torch.stack([
            self.generate_text_embeddings(description)
            for description in descriptions
        ])
        return Embeddings(
            video=embeddings[ModalityType.VISION],
            description=description_embeddings,
        )

    @torch.no_grad()
    def embed_text(self, texts: List[str]) -> torch.Tensor:
        return torch.stack([
            self.generate_text_embeddings(text)
            for text in texts
        ])

    @torch.no_grad()
    async def embed_async(self, descriptions: List[str], video_files: List[BinaryIO]) -> Embeddings:
        inputs = self.get_inputs(descriptions, video_files)  # cannot be async
        embeddings = await run_async(self.imagebind, inputs)
        text_embeddings = torch.stack([
            await run_async(self.generate_text_embeddings, text)
            for text in descriptions
        ])
        return Embeddings(
            video=embeddings[ModalityType.VISION],
            audio=embeddings[ModalityType.AUDIO],
            description=text_embeddings,
        )

    async def embed_text_async(self, texts: List[str]) -> torch.Tensor:
        return await run_async(self.embed_text, texts)
