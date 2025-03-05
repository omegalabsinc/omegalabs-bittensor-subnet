import os
import torch
import torchaudio
import numpy as np

if hasattr(np, "nan"):
    np.NaN = np.nan
    np.NAN = np.nan
from pyannote.audio import Pipeline
import pandas as pd


class CustomDiarizationPipeline:
    def __init__(self, overlap_detection_model_id, diarization_model_id, device="cuda"):
        self.device = torch.device(device)
        self.overlapped_speech_detection_pipeline = Pipeline.from_pretrained(
            overlap_detection_model_id
        ).to(self.device)
        self.diarization_pipeline = Pipeline.from_pretrained(diarization_model_id).to(
            self.device
        )

    def preprocess_audio(self, audio_arr, sr):
        waveform, sample_rate = torch.from_numpy(audio_arr), sr
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply high-pass filter to remove low frequency noise
        waveform = torchaudio.functional.highpass_biquad(
            waveform, sample_rate, cutoff_freq=100
        )

        # Apply noise reduction using spectral subtraction
        spec = torch.stft(
            waveform[0],
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window=torch.hann_window(2048).to(waveform.device),
            return_complex=True,
        )

        # Estimate noise from first few frames
        noise_estimate = torch.mean(torch.abs(spec[:, :50]), dim=1, keepdim=True)

        # Subtract noise estimate and apply soft thresholding
        spec_mag = torch.abs(spec)
        spec_phase = torch.angle(spec)
        spec_mag = torch.maximum(
            spec_mag - 2 * noise_estimate, torch.zeros_like(spec_mag)
        )

        # Reconstruct signal
        spec = spec_mag * torch.exp(1j * spec_phase)
        waveform = torch.istft(
            spec,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window=torch.hann_window(2048).to(waveform.device),
        )
        waveform = waveform.unsqueeze(0)

        # Normalize audio
        waveform = waveform / torch.max(torch.abs(waveform))

        return waveform, sample_rate

    def detect_overlapping_speech_and_run_diarization(self, audio_arr, sr):
        # waveform, sample_rate = self.preprocess_audio(audio_arr, sr)
        waveform, sample_rate = (
            torch.from_numpy(audio_arr).unsqueeze(0).to(torch.float32),
            sr,
        )

        overlapping_segments = self.overlapped_speech_detection_pipeline(
            {"waveform": waveform, "sample_rate": sample_rate}
        )
        diarization = self.diarization_pipeline(
            {"waveform": waveform, "sample_rate": sample_rate}
        )
        diar_segments = []
        overlap_segments = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diar_segments.append((turn.start, turn.end, speaker))

        for speech in overlapping_segments.get_timeline().support():
            overlap_segments.append((speech.start, speech.end, None))

        return overlap_segments, diar_segments

    def remove_overlapping_segments(self, overlap_segments, diar_segments):
        for overlap_segment in overlap_segments:
            overlap_start = overlap_segment[0]
            overlap_end = overlap_segment[1]
            temp_diar_segments = []
            for diar_segment in diar_segments:
                speaker = diar_segment[2]
                start = diar_segment[0]
                end = diar_segment[1]
                if overlap_start < end and overlap_end > end:
                    temp_diar_segments.append((start, overlap_start, speaker))
                elif overlap_start < start and overlap_end > start:
                    temp_diar_segments.append((overlap_end, end, speaker))
                elif overlap_start > start and overlap_end < end:
                    temp_diar_segments.append((start, overlap_start, speaker))
                    temp_diar_segments.append((overlap_end, end, speaker))
                else:
                    temp_diar_segments.append(diar_segment)
            diar_segments = temp_diar_segments
        # Remove any segments that were completely overlapped
        diar_segments = [seg for seg in diar_segments if seg is not None]
        return diar_segments

    def write_segments_to_csv(self, segments, output_file, min_duration=0.5):
        """
        Write the start, end, and duration times of diarization segments to a CSV file using pandas.

        Args:
            segments (list): List of tuples containing (start_time, end_time) for each segment.
            output_file (str): Path to the output CSV file.
        """
        data = []
        for segment in segments:
            start = segment[0]
            end = segment[1]
            if len(segment) > 2:
                speaker = segment[2]
            else:
                speaker = None
            duration = end - start
            if duration >= min_duration:
                data.append(
                    {
                        "Start": start,
                        "End": end,
                        "Duration": duration,
                        "Speaker": speaker,
                    }
                )

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

    def filter_segments_by_duration(self, segments, min_duration=0.7):
        return [
            segment for segment in segments if segment[1] - segment[0] >= min_duration
        ]

    def generate_audio_patches(
        self, audio_arr, sr, segments, output_dir, min_duration=0.5
    ):
        # Load the audio file using pydub
        audio, sr = self.preprocess_audio(audio_arr, sr)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate audio patches for each speaker segment
        for idx, segment in enumerate(segments):
            start_time, end_time, speaker = segment
            duration = end_time - start_time

            # Skip segments shorter than min_duration
            if duration < min_duration:
                continue

            # Calculate start and end times in milliseconds
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)

            # Extract the audio segment
            audio_segment = audio[start_ms:end_ms]

            # Generate output filename
            output_filename = f"{start_ms:07d}.wav"
            output_path = os.path.join(output_dir, output_filename)
            # print(f"Saving {output_path}")

            # Export the audio segment
            audio_segment.export(output_path, format="wav")

        print(f"Audio patches generated and saved in {output_dir}")

    def segments_to_dict(self, segments):
        start_timestamps = [segment[0] for segment in segments]
        end_timestamps = [segment[1] for segment in segments]
        speakers = [segment[2] for segment in segments]
        return {"start": start_timestamps, "end": end_timestamps, "speakers": speakers}

    def process(self, audio_arr, sr, output_path=None):
        overlapping_segments, diar_segments = (
            self.detect_overlapping_speech_and_run_diarization(audio_arr, sr)
        )

        filtered_overlapping_segments = self.filter_segments_by_duration(
            overlapping_segments
        )
        diar_segments = self.remove_overlapping_segments(
            filtered_overlapping_segments, diar_segments
        )
        dataframe = self.segments_to_dict(diar_segments)
        return dataframe
