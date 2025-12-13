from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import queue
import sys
import webrtcvad
import collections

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION = 30  # ms - webrtcvad requires 10, 20, or 30ms frames
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

# VAD configuration
VAD_AGGRESSIVENESS = 1  # 0-3, higher = more aggressive filtering
PADDING_DURATION = 500  # ms - silence before/after speech to keep
NUM_PADDING_FRAMES = int(PADDING_DURATION / FRAME_DURATION)
MIN_SPEECH_FRAMES = 10  # Minimum frames to consider as speech

# Load Whisper model
print("Loading faster-whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Model loaded!")

# Initialize VAD
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

def frame_generator(audio_data):
    """Generate frames from audio data"""
    offset = 0
    while offset + FRAME_SIZE <= len(audio_data):
        yield audio_data[offset:offset + FRAME_SIZE]
        offset += FRAME_SIZE

def transcribe_audio(audio_data):
    """Transcribe using faster-whisper"""
    # Convert float32 to int16
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Transcribe directly from numpy array
    segments, info = model.transcribe(
        audio_int16,
        language="en",
        vad_filter=False  # We're doing our own VAD
    )
    text = " ".join([segment.text for segment in segments])
    return text.strip()

class VADProcessor:
    def __init__(self):
        self.triggered = False
        self.voiced_frames = []
        self.ring_buffer = collections.deque(maxlen=NUM_PADDING_FRAMES)
        self.speech_frames = []
    
    def process_frame(self, frame):
        """
        Process a single audio frame.
        Returns audio data if speech segment completed, None otherwise.
        """
        # Convert float32 frame to int16 for VAD
        frame_int16 = (frame * 32767).astype(np.int16).tobytes()
        
        is_speech = vad.is_speech(frame_int16, SAMPLE_RATE)
        
        if not self.triggered:
            # Not currently in speech segment
            self.ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in self.ring_buffer if speech])
            
            # Start of speech detected
            if num_voiced > 0.8 * self.ring_buffer.maxlen:
                self.triggered = True
                # Add buffered frames to speech
                self.speech_frames.extend([f for f, s in self.ring_buffer])
                self.ring_buffer.clear()
        else:
            # Currently in speech segment
            self.speech_frames.append(frame)
            self.ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
            
            # End of speech detected
            if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                self.triggered = False
                
                # Check if we have enough speech frames
                if len(self.speech_frames) >= MIN_SPEECH_FRAMES:
                    # Return the speech segment
                    audio_segment = np.concatenate(self.speech_frames)
                    self.speech_frames = []
                    self.ring_buffer.clear()
                    return audio_segment
                else:
                    # Too short, discard
                    self.speech_frames = []
                    self.ring_buffer.clear()
        
        return None

print(f"\nRecording... (Press Ctrl+C to stop)")
print("Speak naturally - transcription will happen after each sentence/word")
print(f"VAD Aggressiveness: {VAD_AGGRESSIVENESS} (0=least, 3=most aggressive)")
print("-" * 60)

vad_processor = VADProcessor()
buffer = np.array([], dtype=np.float32)

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        callback=audio_callback, dtype=np.float32):
        while True:
            chunk = audio_queue.get()
            buffer = np.append(buffer, chunk.flatten())
            
            # Process buffer in frames
            while len(buffer) >= FRAME_SIZE:
                frame = buffer[:FRAME_SIZE]
                buffer = buffer[FRAME_SIZE:]
                
                # Process frame through VAD
                speech_segment = vad_processor.process_frame(frame)
                
                if speech_segment is not None:
                    # We have a complete speech segment, transcribe it
                    duration = len(speech_segment) / SAMPLE_RATE
                    print(f"[Speech detected: {duration:.1f}s]")
                    
                    text = transcribe_audio(speech_segment)
                    
                    if text:
                        print(f">>> {text}")
                        print("-" * 60)
                
except KeyboardInterrupt:
    print("\n\nStopped recording.")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)