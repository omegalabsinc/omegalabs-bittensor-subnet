import os
from datasets import load_dataset
from huggingface_hub import login

# Set HF_TOKEN environment variable or pass directly
HF_TOKEN = os.getenv('HF_TOKEN')

# Login to Hugging Face
login(token=HF_TOKEN)

# Load the dataset
dataset = load_dataset("tezuesh/diarization_dataset", use_auth_token=HF_TOKEN)

print(f"Dataset loaded successfully with {len(dataset)} examples")
# Get first row from the dataset
first_row = dataset['train'][0]
print("\nFirst row of dataset:")
# print(first_row)
print("\nKeys in dataset:")
print("\nLength of values in first row:")
for key in first_row.keys():
    if isinstance(first_row[key], list):
        print(f"{key}: {len(first_row[key])}")
    else:
        print(f"{key}: {first_row[key]}")



import librosa
import numpy as np
audio_arr = first_row['audio_array']
print(len(audio_arr), type(audio_arr))
sr = 22050
audio = np.array(audio_arr)
# exit()
print(audio.shape)
import soundfile as sf
youtube_id = first_row['youtube_id']
os.makedirs('Dataset_audios/Original', exist_ok=True)
sf.write(f'Dataset_audios/Original/{youtube_id}.wav', audio, sr)

diar_timestamps_start = first_row['diar_timestamps_start']
diar_timestamps_end = first_row['diar_timestamps_end']
diar_speakers = first_row['diar_speakers']

for start, end, speaker in zip(diar_timestamps_start, diar_timestamps_end, diar_speakers):
    # Calculate start and end samples
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    
    # Extract the clip
    clip = audio[start_sample:end_sample]
    
    # Create output directory if it doesn't exist
    os.makedirs(f'Dataset_audios/Clips/{youtube_id}', exist_ok=True)
    
    # Save the clip with speaker and timestamp info in filename
    clip_filename = f'Dataset_audios/Clips/{youtube_id}/speaker_{speaker}_{start:.2f}-{end:.2f}.wav'
    sf.write(clip_filename, clip, sr)
    

# Create a list to store the diarization data
diarization_data = []
for start, end, speaker in zip(diar_timestamps_start, diar_timestamps_end, diar_speakers):
    diarization_data.append({
        'youtube_id': youtube_id,
        'start_time': start,
        'end_time': end, 
        'speaker': speaker,
        "duration": end - start
    })

# Convert to pandas DataFrame and save as CSV
import pandas as pd
df = pd.DataFrame(diarization_data)
os.makedirs('Dataset_audios/Metadata', exist_ok=True)
df.to_csv(f'Dataset_audios/Metadata/{youtube_id}_diarization.csv', index=False)
