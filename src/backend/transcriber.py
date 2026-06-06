"""
transcriber.py - Complete rewrite
Uses imageio-ffmpeg to convert webm → wav
Then feeds wav directly to Whisper using numpy array
No system ffmpeg PATH needed at all.
"""

import os
import numpy as np
import subprocess
import tempfile


class Transcriber:
    def __init__(self, model_size: str = "base"):
        print(f"[STT] Loading Whisper {model_size}...")
        import whisper
        self.model      = whisper.load_model(model_size)
        self.ffmpeg_exe = self._get_ffmpeg()
        print(f"[STT] Ready. ffmpeg={self.ffmpeg_exe is not None}")

    def _get_ffmpeg(self):
        """Get ffmpeg path from imageio-ffmpeg."""
        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"[STT] ffmpeg found: {path}")
            return path
        except Exception as e:
            print(f"[STT] imageio-ffmpeg not found: {e}")
            return None

    def _webm_to_numpy(self, webm_path: str) -> np.ndarray:
        """
        Convert webm/any audio to numpy array at 16kHz using ffmpeg.
        This is what Whisper needs — raw float32 samples.
        """
        if not self.ffmpeg_exe:
            return None
        try:
            cmd = [
                self.ffmpeg_exe,
                "-nostdin",
                "-threads", "0",
                "-i", webm_path,
                "-f", "s16le",       # signed 16-bit little-endian PCM
                "-ac", "1",          # mono
                "-acodec", "pcm_s16le",
                "-ar", "16000",      # 16kHz
                "-"                  # output to stdout
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120
            )
            if result.returncode != 0:
                err = result.stderr.decode(errors='ignore')
                print(f"[STT] ffmpeg error: {err[:300]}")
                return None
            if len(result.stdout) < 1000:
                print(f"[STT] Audio too short: {len(result.stdout)} bytes")
                return None
            # Convert bytes to float32
            audio = np.frombuffer(result.stdout, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0
            duration = len(audio) / 16000
            print(f"[STT] Audio decoded: {duration:.1f}s, {len(audio)} samples")
            return audio
        except subprocess.TimeoutExpired:
            print("[STT] ffmpeg timed out")
            return None
        except Exception as e:
            print(f"[STT] Conversion error: {e}")
            return None

    def transcribe_file(self, file_path: str) -> str:
        """Transcribe any audio/video file."""
        if not file_path:
            return ""

        # Handle both webm and wav paths
        if not os.path.exists(file_path):
            # Try wav version
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

        # Convert to numpy array
        audio = self._webm_to_numpy(file_path)
        if audio is None or len(audio) < 1600:  # less than 0.1 seconds
            print("[STT] No usable audio")
            return ""

        try:
            import whisper
            result = self.model.transcribe(
                audio,                          # pass numpy array directly
                language    = "en",
                fp16        = False,            # CPU mode
                beam_size   = 1,               # fast
                best_of     = 1,               # fast
                temperature = 0,               # deterministic
                condition_on_previous_text = False,
                verbose     = False,
                no_speech_threshold = 0.5,
                compression_ratio_threshold = 2.8,
            )
            text = result["text"].strip()
            if text:
                print(f"[STT] ✅ '{text[:120]}'")
            else:
                print("[STT] Empty transcript")
            return text
        except Exception as e:
            print(f"[STT] Whisper error: {e}")
            return ""

    def transcribe_all(self, recordings):
        for rec in recordings:
            q = rec.get("question", {})
            qid = q.get("id", "?") if isinstance(q, dict) else "?"
            print(f"[STT] Q{qid}...")
            rec["transcript"] = self.transcribe_file(rec.get("audio_path", ""))
        return recordings