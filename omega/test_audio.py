from omega.video_utils import get_audio_bytes
import base64

audio_bytes = get_audio_bytes("test_video.mp4")
print(audio_bytes)

# Save audio bytes to a WAV file
with open("output_audio.wav", "wb") as f:
    f.write(audio_bytes)

audio_bytes_b64 = base64.b64encode(audio_bytes).decode("utf-8")
print(audio_bytes_b64)
# Save base64 encoded audio to file
with open("output_audio_b64.txt", "w") as f:
    f.write(audio_bytes_b64)
