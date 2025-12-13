import vosk
import pyaudio
import json

#vosk_model = "models/vosk-model-small-en-us-0.15"
vosk_model = "models/vosk-model-en-us-0.22"
#vosk_model = "models/vosk-model-en-us-0.22-lgraph"
#vosk_model = "models/vosk-model-small-cs-0.4-rhasspy"
model = vosk.Model(vosk_model)
rec = vosk.KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000)
stream.start_stream()

print("Listening...")
while True:
    data = stream.read(8000, exception_on_overflow=False)
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        print("Final:", result['text'])