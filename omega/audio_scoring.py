if hasattr(np, 'nan'):
    np.NaN = np.nan
    np.NAN = np.nan
from pyannote.audio import Pipeline
import librosa
import os
import dotenv
import pandas as pd
import numpy as np
import torch


dotenv.load_dotenv()

class AudioScore:
    def __init__(self, device="cuda"):

        self.device = torch.device(device)

        # Load the audio file   
        self.pipeline = Pipeline.from_pretrained("salmanshahid/vad").to(self.device)
        

        self.steepness = 5
        self.midpoint = 0.3
    
    

    def speech_content_score(self, audio_arr, sr):
        self.total_duration = librosa.get_duration(y=audio_arr, sr=sr)
        output = self.pipeline({"waveform": torch.from_numpy(audio_arr.astype(np.float32)).unsqueeze(0).to(self.device), "sample_rate": sr})

        self.total_speech_duration = 0   
        for speech in output.get_timeline().support():
            self.total_speech_duration += speech.end - speech.start

        ratio =  self.total_speech_duration / self.total_duration


        return ratio
    
    def speaker_dominance_score(self, timestamps_start, timestamps_end, speakers, dominance_threshold=0.7):
        if timestamps_start is None:
            self.rttm_data = None
            return 0
        self.rttm_data = pd.DataFrame({
            'start': timestamps_start,
            'end': timestamps_end,
            'speaker': speakers
        })

        # If there's only one speaker, return 0 since dominance is expected
        if len(set(speakers)) == 1:
            return 0

        # Calculate total duration for each speaker
        speaker_durations = {}
        for _, row in self.rttm_data.iterrows():
            speaker = row['speaker']
            duration = row['end'] - row['start']
            if speaker in speaker_durations:
                speaker_durations[speaker] += duration
            else:
                speaker_durations[speaker] = duration
        max_time = max(speaker_durations.values())
        min_time = min(speaker_durations.values())

        return 1 - (max_time - min_time) / self.total_duration
        

    def background_noise_score(self, audio_arr, sr, noise_threshold=0.1):
        # Load audio and calculate SNR
        self.audio = audio_arr
        self.sr = sr
        
        # Calculate signal power
        signal_power = np.mean(self.audio**2)
        
        # Estimate noise power (using the lowest 10% of frame energies as noise estimate)
        frame_length = int(0.025 * self.sr)  # 25ms frames
        frames = librosa.util.frame(self.audio, frame_length=frame_length, hop_length=frame_length)
        frame_energies = np.mean(frames**2, axis=0)
        noise_power = np.mean(np.percentile(frame_energies, 10))
        
        # Calculate SNR in dB
        if noise_power == 0:
            snr = 100  # High SNR for very clean signal
        else:
            snr = 10 * np.log10(signal_power / noise_power)
            
        # Convert SNR to penalty score (higher SNR = lower penalty)
        return 1 - max(0, 1 - (snr / 50))  # Normalize to 0-1 range, assuming 50dB as reference
    
    def unique_speakers_error(self, speakers):
        unique_speakers = len(set(speakers))
        if unique_speakers == 2:
            return 1
        elif unique_speakers == 1 or unique_speakers == 0 or unique_speakers > 4:
            return 0
        else:
            return 1/(unique_speakers-1)

    def total_score(self, audio_arr, sr, timestamps_start, timestamps_end, speakers):
        audio_arr = np.array(audio_arr)
        timestamps_start = np.array(timestamps_start)
        timestamps_end = np.array(timestamps_end)
        # speakers = torch.tensor(speakers)
        speech_content_score = self.speech_content_score(audio_arr, sr)
        speaker_dominance_score = self.speaker_dominance_score(timestamps_start, timestamps_end, speakers)
        background_noise_score = self.background_noise_score(audio_arr, sr)
        return {
            "speech_content_score": speech_content_score, 
            "speaker_dominance_score": speaker_dominance_score, 
            "background_noise_score": background_noise_score,
            "unique_speakers_error": self.unique_speakers_error(speakers),
        }


if __name__ == "__main__":    

    from datasets import load_dataset
    import huggingface_hub


    repo_id = "diarizers-community/voxconverse"

    ds = load_dataset(repo_id, split="test", cache_dir="/workspace/tezuesh/voxconverse/data_cache")

    ds = next(ds.shuffle().iter(batch_size=64))
    audio_arr = ds['audio'][0]['array']
    sr = ds['audio'][0]['sampling_rate']
    timestamps_start = ds['timestamps_start'][0]
    timestamps_end = ds['timestamps_end'][0]
    speakers = ds['speakers'][0]


    # # Save test audio to WAV file
    import soundfile as sf
    
    output_audio_path = 'test_audio.wav'
    sf.write(output_audio_path, audio_arr, sr)
    print(f"Saved test audio to {output_audio_path}")
    # Create a DataFrame with timestamps and speakers
    import pandas as pd
    
    df = pd.DataFrame({
        'start': timestamps_start,
        'end': timestamps_end,
        'speaker': speakers
    })
    
    # Save to CSV file
    output_path = 'speaker_timestamps.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved speaker timestamps to {output_path}")
    audio_score = AudioScore()
    
    score = audio_score.total_score(audio_arr, sr, timestamps_start, timestamps_end, speakers)
    print(score)
