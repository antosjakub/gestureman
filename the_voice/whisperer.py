from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import queue
import sys
from scipy.io import wavfile
import tempfile
import os

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 3  # Shorter chunks with faster-whisper
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

# Load model (much faster than standard Whisper)
print("Loading faster-whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Model loaded!")

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

def transcribe_audio(audio_data):
    """Transcribe using faster-whisper"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        wavfile.write(temp_path, SAMPLE_RATE, audio_data)
    
    try:
        segments, info = model.transcribe(temp_path, language="en", vad_filter=True)
        text = " ".join([segment.text for segment in segments])
        return text
    finally:
        os.unlink(temp_path)

print(f"\nRecording... (Press Ctrl+C to stop)")
print(f"Transcribing every {CHUNK_DURATION} seconds\n")
print("-" * 60)

buffer = np.array([], dtype=np.float32)

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        callback=audio_callback, dtype=np.float32):
        while True:
            chunk = audio_queue.get()
            buffer = np.append(buffer, chunk.flatten())
            
            if len(buffer) >= CHUNK_SIZE:
                audio_to_transcribe = buffer[:CHUNK_SIZE]
                buffer = buffer[CHUNK_SIZE:]
                
                audio_int16 = (audio_to_transcribe * 32767).astype(np.int16)
                text = transcribe_audio(audio_int16)
                
                if text.strip():
                    print(f">>> {text}")
                    print("-" * 60)
                
except KeyboardInterrupt:
    print("\n\nStopped recording.")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)