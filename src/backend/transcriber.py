import os

class Transcriber:
    def __init__(self, model_size: str = "small"):
        print(f"[STT] Loading faster-whisper '{model_size}' (cpu/int8)...")
        from faster_whisper import WhisperModel
        # int8 = fast + low memory on CPU; first run downloads the model once.
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("[STT] Ready.")

    def transcribe_file(self, file_path: str) -> str:
        if not file_path:
            return ""

        if not os.path.exists(file_path):
            wav_path = file_path.replace(".webm", ".wav")
            if os.path.exists(wav_path):
                file_path = wav_path
            else:
                print(f"[STT] File not found: {file_path}")
                return ""

        file_size = os.path.getsize(file_path)
        print(f"[STT] Transcribing {os.path.basename(file_path)} ({file_size} bytes)")
        if file_size < 500:
            print("[STT] File too small, skipping")
            return ""

        try:
            segments, info = self.model.transcribe(
                file_path,
                language="en",
                beam_size=5,                     
                vad_filter=True,                 
                vad_parameters={"min_silence_duration_ms": 500}, 
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                temperature=0,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if text:
                print(f"[STT] OK '{text[:120]}'")
            else:
                print("[STT] Empty transcript (likely silence)")
            return text
        except Exception as e:
            print(f"[STT] faster-whisper error: {e}")
            return ""

    def transcribe_all(self, recordings):
        for rec in recordings:
            q = rec.get("question", {})
            qid = q.get("id", "?") if isinstance(q, dict) else "?"
            print(f"[STT] Q{qid}...")
            rec["transcript"] = self.transcribe_file(rec.get("audio_path", ""))
        return recordings
