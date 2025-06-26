from faster_whisper import WhisperModel
import torch
import io
import librosa
import soundfile as sf
import tempfile

def convert_bytes_to_array(audio_bytes):
    audio_bytes = io.BytesIO(audio_bytes)
    audio, sample_rate = librosa.load(audio_bytes, sr=None)  # sr=None, damit Original-Sample-Rate beibehalten wird
    return audio, sample_rate

def transcribe_audio(audio_bytes):
    model_size = "small"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")

    # Bytes in Audiodaten konvertieren
    audio_array, sample_rate = convert_bytes_to_array(audio_bytes)

    # Temporäre WAV-Datei erstellen
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
        sf.write(tmpfile.name, audio_array, sample_rate)
        segments, info = model.transcribe(tmpfile.name)

    # Transkript zusammenbauen
    transcript = "".join([segment.text for segment in segments])
    return transcript