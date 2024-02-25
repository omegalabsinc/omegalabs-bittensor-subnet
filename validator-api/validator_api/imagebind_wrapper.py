import tempfile
from typing import List, BinaryIO

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from moviepy.editor import VideoFileClip
from pydantic import BaseModel
import torch

from omega.miner_utils import download_video_from_id
from omega.protocol import VideoMetadata


class Embeddings(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    video: torch.Tensor
    audio: torch.Tensor
    description: torch.Tensor


def copy_audio(video_path: str) -> BinaryIO:
    # Lazy load the video file
    temp_audiofile = tempfile.NamedTemporaryFile(suffix=".aac")
    video_clip = VideoFileClip(video_path)
    
    # Extract the audio from the video
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(temp_audiofile.name, codec='aac')
    
    # Close the clips to release resources
    audio_clip.close()
    video_clip.close()

    return temp_audiofile


class ImageBind:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.imagebind = imagebind_model.imagebind_huge(pretrained=True)
        self.imagebind.eval()
        self.imagebind.to(self.device)

    @torch.no_grad()
    def embed(self, videos: List[VideoMetadata], video_files: List[BinaryIO]) -> Embeddings:
        audio_files = [copy_audio(video_file.name) for video_file in video_files]
        audio_filepaths = [audio_file.name for audio_file in audio_files]
        video_filepaths = [video_file.name for video_file in video_files]
        try:
            video_data = data.load_and_transform_video_data(video_filepaths, self.device)
            audio_data = data.load_and_transform_audio_data(audio_filepaths, self.device)
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text([video.description for video in videos], self.device),
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


if __name__ == "__main__":
    video_id = "dQw4w9WgXcQ"
    video_metadata = download_video_from_id(video_id)
    model = ImageBind()
    emb = model.embed(video_metadata)
    print(emb)
