from pyannote.audio import Pipeline
import librosa
import os
import dotenv
import pandas as pd
import numpy as np
import torch
from scipy.stats import entropy

dotenv.load_dotenv()

HUGGINGFACE_AUTH_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN")

class VoicePenalty:
    def __init__(self, device="cuda"):

        self.device = torch.device(device)

        # Load the audio file   
        self.pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token=HUGGINGFACE_AUTH_TOKEN).to(self.device)
        

        self.steepness = 5
        self.midpoint = 0.3
    
    # def calculate_energy(self):
    #     """
    #     Calculate the energy of an audio signal.

    #     Args:
    #         audio (array-like): The audio signal as a NumPy array.

    #     Returns:
    #         float: The energy of the audio signal.
    #     """
    #     self.audio, self.sr = librosa.load(self.audio_path)
    #     # Calculate energy for each second
    #     frame_length = self.sr  # One second worth of samples
    #     num_frames = int(len(self.audio) / frame_length)
    #     energies = []
        
    #     for i in range(num_frames):
    #         start = i * frame_length
    #         end = start + frame_length
    #         frame = self.audio[start:end]
    #         frame_energy = np.sum(frame**2)
    #         energies.append(frame_energy)
            
    #     # Handle the last partial frame if it exists
    #     if len(self.audio) % frame_length != 0:
    #         last_frame = self.audio[num_frames * frame_length:]
    #         last_energy = np.sum(last_frame**2)
    #         energies.append(last_energy)
            
    #     energies = np.array(energies)
    #     return np.mean(energies)  # Return average energy across all seconds


    def speech_content_penalty(self, audio_path):
        self.total_duration = librosa.get_duration(path=audio_path)
        output = self.pipeline(audio_path)

        self.total_speech_duration = 0   
        for speech in output.get_timeline().support():
            self.total_speech_duration += speech.end - speech.start
        print(f"Total speech duration: {self.total_speech_duration}")
        
        ratio =  self.total_speech_duration / self.total_duration
        return 1 - (1 / (1 + np.exp(-self.steepness * (  - self.midpoint)))), 1-ratio
    
    def speaker_dominance_penalty(self, rttm_path, dominance_threshold=0.7):
        if rttm_path is None:
            self.rttm_data = None
            return 1
        self.rttm_data = pd.read_csv(rttm_path, sep=' ', header=None, 
                                names=['Type', 'File', 'Channel', 'Start', 'Duration', 'Orthography', 'Confidence', 'Speaker', 'NA1', 'NA2'])

        # Calculate total duration for each speaker
        speaker_durations = {}
        for _, row in self.rttm_data.iterrows():
            speaker = row['Speaker']
            duration = row['Duration']
            if speaker in speaker_durations:
                speaker_durations[speaker] += duration
            else:
                speaker_durations[speaker] = duration
        max_speaking_time = max(speaker_durations.values())
        print(f"Max speaking time: {max_speaking_time}")
        max_speaking_proportion = max_speaking_time / self.total_duration
        penalty = max(0, max_speaking_proportion - dominance_threshold)/(1 - dominance_threshold)
        return penalty
    
    # def synthetic_audio_penalty(self, min_segment_duration=0.5, max_segment_duration=30.0, 
    #                           expected_transition_rate=0.2, entropy_weight=0.3):
    #     """
    #     Calculates penalty score for potentially synthetic audio based on:
    #     1. Unnatural segment durations
    #     2. Speaker transition patterns
    #     3. Speaker distribution entropy
        
    #     Args:
    #         min_segment_duration: Minimum natural segment duration in seconds
    #         max_segment_duration: Maximum natural segment duration in seconds
    #         expected_transition_rate: Expected natural rate of speaker transitions
    #         entropy_weight: Weight given to entropy component
            
    #     Returns:
    #         float: Penalty score between 0-1, higher indicates more synthetic
    #     """
    #     if self.rttm_data is None:
    #         return 1
            
    #     # Get segment durations and transitions
    #     segment_durations = self.rttm_data['Duration'].values
    #     speakers = self.rttm_data['Speaker'].values
        
    #     # Duration penalty - penalize segments outside natural range
    #     duration_penalties = []
    #     for dur in segment_durations:
    #         if dur < min_segment_duration or dur > max_segment_duration:
    #             duration_penalties.append(1.0)
    #         else:
    #             duration_penalties.append(0.0)
    #     duration_penalty = np.mean(duration_penalties)
        
    #     # Transition penalty - compare to expected natural rate
    #     transitions = sum(speakers[i] != speakers[i+1] for i in range(len(speakers)-1))
    #     transition_rate = transitions / len(speakers)
    #     transition_penalty = abs(transition_rate - expected_transition_rate)
        
    #     # Entropy penalty - unnatural speaker distribution
    #     speaker_durations = {}
    #     for _, row in self.rttm_data.iterrows():
    #         spk = row['Speaker']
    #         dur = row['Duration']
    #         speaker_durations[spk] = speaker_durations.get(spk, 0) + dur
            
    #     total_duration = sum(speaker_durations.values())
    #     probs = [d/total_duration for d in speaker_durations.values()]
    #     entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    #     max_entropy = np.log2(len(speaker_durations))
    #     entropy_penalty = 1 - (entropy / max_entropy)
    #     print("Inside synthetic audio penalty")
    #     print(f"duration_penalty: {duration_penalty}, transition_penalty: {transition_penalty}, entropy_penalty: {entropy_penalty}")
        
    #     # Combine penalties with weights
    #     final_penalty = duration_penalty + \
    #                     transition_penalty + \
    #                    entropy_weight * entropy_penalty
                       
    #     return min(1.0, final_penalty)
    
    # def speaker_fabrication_penalty(self, target_variance=0.1, spread=0.05):
    #     """
    #     Penalizes fabricated audio based on speaker duration variance, transition rate,
    #     and entropy of speaker distribution.

    #     Args:
    #         target_variance (float): Desired variance for natural speaker durations
    #             - Lower values (~0.05-0.1): Expect more balanced speaking times
    #             - Higher values (~0.2-0.3): Allow more uneven speaking distributions
    #         spread (float): Controls how quickly penalty increases when variance differs from target
    #             - Lower values: Stricter penalty
    #             - Higher values: More lenient penalty

    #     Returns:
    #         float: Combined penalty score. Higher indicates more fabrication likelihood.
    #     """
    #     # Convert RTTM data to speaker segments
    #     if self.rttm_data is None:
    #         return 1
    #     speaker_segments = []
    #     for _, row in self.rttm_data.iterrows():
    #         speaker_segments.append((row['Speaker'], row['Duration']))

    #     # Group segments by speaker
    #     speaker_durations = {}
    #     for speaker_id, duration in speaker_segments:
    #         if speaker_id in speaker_durations:
    #             speaker_durations[speaker_id] += duration
    #         else:
    #             speaker_durations[speaker_id] = duration

    #     # Convert durations to list for each speaker
    #     durations = np.array(list(speaker_durations.values()))

    #     # Metric 1: Speaker Duration Variance Penalty
    #     # Calculate variance of speaker durations to detect unnatural speaking patterns
    #     variance = np.var(durations)
    #     duration_penalty = np.exp(-((variance - target_variance) ** 2) / (2 * spread ** 2))

    #     # Metric 2: Speaker Transition Rate Penalty
    #     # Count speaker transitions (changes between consecutive segments)
    #     transitions = sum(speaker_segments[i][0] != speaker_segments[i+1][0] for i in range(len(speaker_segments)-1))
    #     transition_rate = transitions / len(speaker_segments)
    #     transition_penalty = 1 - transition_rate  # Penalizes very low or perfect transition rates

    #     # Metric 3: Entropy of Speaker Distribution
    #     total_duration = sum(durations)
    
    #     if total_duration == 0:
    #         return 1.0  # Avoid division by zero if there are no durations
        
    #     # Normalize counts to probabilities
    #     speaker_probabilities = durations / total_duration
    #     distribution_entropy = entropy(speaker_probabilities)

    #     # Calculate maximum entropy for the number of speakers
    #     num_speakers = len(speaker_durations)
    #     max_entropy = np.log(num_speakers) if num_speakers > 1 else 1  # Handle case with only one speaker

    #     # Normalize entropy penalty (0 for perfect balance, 1 for perfect imbalance)
    #     entropy_penalty = distribution_entropy / max_entropy  # Ensure it scales between 0 and 1

    #     # Combine penalties with weights
    #     combined_penalty = 0.4 * duration_penalty + 0.3 * transition_penalty + 0.3 * entropy_penalty
    #     print(f"duration_penalty: {duration_penalty}, transition_penalty: {transition_penalty}, entropy_penalty: {entropy_penalty}")
    #     return combined_penalty
    

    def background_noise_penalty(self, audio_path, noise_threshold=0.1):
        # Load audio and calculate SNR
        self.audio, self.sr = librosa.load(audio_path)
        
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
        return max(0, 1 - (snr / 50))  # Normalize to 0-1 range, assuming 50dB as reference

    

    def total_penalty(self, audio_path, rttm_path):
        speech_content_penalty, ratio = self.speech_content_penalty(audio_path)
        speaker_dominance_penalty = self.speaker_dominance_penalty(rttm_path)
        # speaker_fabrication_penalty = self.speaker_fabrication_penalty()
        background_noise_penalty = self.background_noise_penalty(audio_path)
        # synthetic_audio_penalty = self.synthetic_audio_penalty()
        return {"speech_content_penalty": speech_content_penalty, 
                "speaker_dominance_penalty": speaker_dominance_penalty, 
                # "speaker_fabrication_penalty": speaker_fabrication_penalty, 
                "speech_ratio": ratio, 
                "background_noise_penalty": background_noise_penalty,
                # "synthetic_audio_penalty": synthetic_audio_penalty
                }

    

    

    

    


if __name__ == "__main__":    
    rttm_root = "/workspace/tezuesh/omega-v2v/dataset/Results/Revai_Diarization/"  
    tags = os.listdir(rttm_root)[:5]
    audio_root = "/workspace/tezuesh/datasets/voxconverse_test_wav/"
    # Example usage
    gt_paths = [os.path.join(audio_root, tag + ".wav") for tag in tags]
    rttm_paths = [os.path.join(rttm_root, tag, "output.rttm") for tag in tags]






    noise_samples = [os.path.join('/workspace/tezuesh/omega-v2v/.noise_samples', f) for f in os.listdir('/workspace/tezuesh/omega-v2v/.noise_samples')]

    # sample_path = gt_paths[0]
    # sample_path ="/workspace/tezuesh/omega-v2v/.podcasts/Fresh_Air_Best_Of_Jeremy_Strong_Will_Harpers_Roadtrip_Across_America_sample.mp3"
    # sample_path = "/workspace/tezuesh/omega-v2v/dataset/concatenated_segments.wav"
    # rttm_path = "/workspace/tezuesh/omega-v2v/dataset/concatenated_segments.rttm"

    # audio_root = "./.sn24_test"
    # sample_paths = [os.path.join(audio_root, f) for f in os.listdir(audio_root)]


    # # sample_path = gt_paths[0]
    # rttm_path = rttm_paths[0]

    # print("-"*100)
    # all_penalties = {}
    voice_penalty = VoicePenalty()
    # for sample_path, rttm_path in zip(gt_paths, rttm_paths):
    #     penalty = voice_penalty.total_penalty(sample_path, rttm_path)
    #     print(sample_path)
    #     print(penalty)
    #     all_penalties[sample_path] = penalty
    
    # # Convert dictionary to DataFrame and save to CSV
    # df = pd.DataFrame.from_dict(all_penalties, orient='index').reset_index()
    # df.columns = ['filename'] + list(df.columns[1:])  # Rename first column to 'filename'
    # df.to_csv('gt_set_penalties.csv', index=False)

    aud_path = noise_samples[0]
    penalty = voice_penalty.total_penalty(aud_path, None)
    print(penalty)
