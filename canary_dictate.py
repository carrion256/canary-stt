#!/usr/bin/env python3
"""
Canary Dictate — GPU-accelerated speech-to-text daemon using NVIDIA Canary Qwen 2.5B.

Architecture:
  - Loads model into VRAM on startup and keeps it hot
  - SIGUSR1: Toggle recording (start/stop)
  - SIGUSR2: Cancel current recording without transcribing
  - Records from default PipeWire input (Focusrite Scarlett Solo)
  - Transcribes via Canary Qwen 2.5B on GPU
  - Types result into active window via wtype
  - HTTP API: OpenAI-compatible /v1/audio/transcriptions endpoint

Usage:
  # Start daemon:
  canary-dictate.py

  # Toggle recording (from keybind):
  pkill -USR1 -f canary-dictate.py

  # Cancel recording:
  pkill -USR2 -f canary-dictate.py

  # Transcribe via API:
  curl -X POST http://localhost:8393/v1/audio/transcriptions \\
    -F file=@audio.wav -F model=canary-qwen-2.5b
"""

import signal
import sys
import os
import time
import threading
import tempfile
import subprocess
import logging
import wave
import struct
import uuid

import torch
from pathlib import Path
from enum import Enum, auto

import numpy as np
import sounddevice as sd

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("canary-dictate")

# ── Configuration ──────────────────────────────────────────────────────────

MODEL_NAME = "nvidia/canary-qwen-2.5b"
SAMPLE_RATE = 16000  # Canary expects 16kHz mono
CHANNELS = 1
MAX_RECORD_SECS = 600  # 10 minute max (user does long rambles)
SILENCE_TIMEOUT = 4.0  # auto-stop after 4s of silence
SILENCE_THRESHOLD = 0.01  # RMS below this = silence
DTYPE = np.float32
API_HOST = "0.0.0.0"
API_PORT = 8393

# Path to cache audio chunks
AUDIO_DIR = Path(tempfile.gettempdir()) / "canary-dictate"
AUDIO_DIR.mkdir(exist_ok=True)


class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()


class CanaryDictate:
    def __init__(self):
        self.state = State.IDLE
        self.lock = threading.Lock()
        self.transcribe_lock = threading.Lock()
        self.audio_chunks: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.model = None
        self.last_speech_time = 0.0
        self._silence_monitor: threading.Thread | None = None

    # ── Model Loading ──────────────────────────────────────────────────

    def load_model(self):
        log.info("Loading %s onto GPU (bf16)...", MODEL_NAME)
        t0 = time.time()
        torch.set_float32_matmul_precision("high")
        from nemo.collections.speechlm2.models import SALM

        self.model = SALM.from_pretrained(MODEL_NAME)
        self.model = self.model.bfloat16().cuda()
        self.model.eval()
        elapsed = time.time() - t0
        log.info(
            "Model loaded in %.1fs (%.2f GB VRAM). Ready.",
            elapsed,
            torch.cuda.memory_allocated() / 1e9,
        )

    # ── Audio Recording ────────────────────────────────────────────────

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            log.warning("Audio status: %s", status)
        self.audio_chunks.append(indata.copy())
        rms = np.sqrt(np.mean(indata**2))
        if rms > SILENCE_THRESHOLD:
            self.last_speech_time = time.time()

    def start_recording(self):
        """Start capturing audio from default input."""
        with self.lock:
            if self.state != State.IDLE:
                log.warning("Cannot start recording in state %s", self.state.name)
                return

            self.audio_chunks = []
            self.last_speech_time = time.time()
            self.state = State.RECORDING
            log.info(
                "Recording started. Speak now (auto-stops after %.0fs silence)...",
                SILENCE_TIMEOUT,
            )

            self._notify("Recording started")
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self._audio_callback,
                blocksize=int(SAMPLE_RATE * 0.1),
            )
            self.stream.start()
            self._silence_monitor = threading.Thread(
                target=self._watch_silence, daemon=True
            )
            self._silence_monitor.start()

    def stop_recording(self) -> np.ndarray | None:
        """Stop recording and return the captured audio as numpy array."""
        with self.lock:
            if self.state != State.RECORDING:
                log.warning("Not recording, nothing to stop.")
                return None

            self.state = State.TRANSCRIBING
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

        if not self.audio_chunks:
            log.warning("No audio captured.")
            with self.lock:
                self.state = State.IDLE
            return None

        audio = np.concatenate(self.audio_chunks, axis=0).flatten()
        duration = len(audio) / SAMPLE_RATE
        log.info("Recording stopped. Captured %.1fs of audio.", duration)
        return audio

    def _watch_silence(self):
        while self.state == State.RECORDING:
            time.sleep(0.25)
            elapsed_silence = time.time() - self.last_speech_time
            if elapsed_silence >= SILENCE_TIMEOUT:
                total_audio = sum(len(c) for c in self.audio_chunks) / SAMPLE_RATE
                if total_audio < 0.5:
                    continue
                log.info(
                    "Auto-stopping after %.1fs of silence (%.1fs total audio).",
                    elapsed_silence,
                    total_audio,
                )
                audio = self.stop_recording()
                if audio is not None:
                    self._transcribe_and_type(audio)
                return

    def cancel_recording(self):
        """Cancel recording without transcribing."""
        with self.lock:
            if self.state != State.RECORDING:
                log.info("Not recording, nothing to cancel.")
                return

            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            self.audio_chunks = []
            self.state = State.IDLE
            log.info("Recording cancelled.")
            self._notify("Recording cancelled")

    # ── Transcription ──────────────────────────────────────────────────

    def transcribe_file(self, wav_path: Path) -> tuple[str, float]:
        """Transcribe a WAV file. Returns (text, audio_duration).

        Thread-safe: serializes GPU access via transcribe_lock.
        Does NOT manage self.state — caller is responsible.
        """
        with self.transcribe_lock:
            t0 = time.time()
            try:
                # Read duration from WAV header
                with wave.open(str(wav_path), "r") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    audio_duration = frames / rate

                with torch.inference_mode():
                    answer_ids = self.model.generate(
                        prompts=[
                            [
                                {
                                    "role": "user",
                                    "content": f"Transcribe the following: {self.model.audio_locator_tag}",
                                    "audio": [str(wav_path)],
                                }
                            ]
                        ],
                        max_new_tokens=1024,
                    )
                    text = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
                elapsed = time.time() - t0
                rtf = elapsed / audio_duration if audio_duration > 0 else 0
                log.info(
                    "Transcribed in %.2fs (%.1fx realtime): %s",
                    elapsed,
                    1.0 / rtf if rtf > 0 else 0,
                    text[:100] + "..." if len(text) > 100 else text,
                )
                return text.strip(), audio_duration
            except Exception as e:
                log.error("Transcription failed: %s", e, exc_info=True)
                return "", 0.0

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio from local recording. Manages state."""
        wav_path = AUDIO_DIR / "recording.wav"
        self._save_wav(audio, wav_path)
        try:
            text, _ = self.transcribe_file(wav_path)
            return text
        finally:
            with self.lock:
                self.state = State.IDLE
            wav_path.unlink(missing_ok=True)

    def _save_wav(self, audio: np.ndarray, path: Path):
        """Save float32 numpy audio to 16-bit WAV file."""
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

    # ── Text Output ────────────────────────────────────────────────────

    def type_text(self, text: str):
        """Type text into the active window using wtype."""
        if not text:
            log.warning("Empty transcription, nothing to type.")
            self._notify("Empty transcription")
            return

        log.info("Typing %d chars via wtype...", len(text))
        # Wait for user to release modifier keys (SUPER+D keybind)
        time.sleep(1.0)
        try:
            # wtype reads from argument
            subprocess.run(["wtype", "--", text], check=True, timeout=10)
            self._notify("Transcription typed")
        except FileNotFoundError:
            log.error("wtype not found! Install: sudo pacman -S wtype")
        except subprocess.TimeoutExpired:
            log.error("wtype timed out.")
        except subprocess.CalledProcessError as e:
            log.error("wtype failed: %s", e)

    # ── Notifications ──────────────────────────────────────────────────

    def _notify(self, message: str):
        """Send desktop notification via notify-send."""
        try:
            subprocess.Popen(
                ["notify-send", "-t", "2000", "-a", "Canary Dictate", message],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass  # notify-send not available, no big deal

    # ── Signal-driven Toggle ───────────────────────────────────────────

    def toggle(self):
        """Toggle recording on/off. On stop, transcribes and types."""
        if self.state == State.IDLE:
            self.start_recording()
        elif self.state == State.RECORDING:
            audio = self.stop_recording()
            if audio is not None:
                # Run transcription in a thread so signals aren't blocked
                t = threading.Thread(target=self._transcribe_and_type, args=(audio,))
                t.daemon = True
                t.start()
        elif self.state == State.TRANSCRIBING:
            log.info("Already transcribing, please wait...")

    def _transcribe_and_type(self, audio: np.ndarray):
        """Transcribe audio and type the result."""
        text = self.transcribe(audio)
        if text:
            self.type_text(text)


# ── HTTP API (OpenAI-compatible) ──────────────────────────────────────────


def create_api(dictate: CanaryDictate):
    """Create FastAPI app with OpenAI-compatible transcription endpoint."""
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import JSONResponse, PlainTextResponse

    app = FastAPI(
        title="Canary Dictate",
        description="OpenAI-compatible speech-to-text API backed by NVIDIA Canary Qwen 2.5B",
        version="1.0.0",
    )

    def _convert_to_wav(input_path: Path, output_path: Path) -> bool:
        """Convert any audio format to 16kHz mono WAV using ffmpeg."""
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_path),
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    str(CHANNELS),
                    "-sample_fmt",
                    "s16",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as e:
            log.error("ffmpeg conversion failed: %s", e)
            return False

    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "canary-qwen-2.5b",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "nvidia",
                }
            ],
        }

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form("canary-qwen-2.5b"),
        language: str = Form("en"),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
    ):
        """OpenAI-compatible audio transcription endpoint.

        Accepts: multipart/form-data with audio file.
        Formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
        Returns: JSON with transcribed text.
        """
        if dictate.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        # Save uploaded file
        request_id = uuid.uuid4().hex[:8]
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        upload_path = AUDIO_DIR / f"upload_{request_id}{suffix}"
        wav_path = AUDIO_DIR / f"upload_{request_id}_out.wav"

        try:
            # Write upload to disk
            content = await file.read()
            upload_path.write_bytes(content)
            log.info(
                "API: Received %s (%d bytes) from %s",
                suffix,
                len(content),
                file.filename,
            )

            # Convert to WAV if needed
            if suffix.lower() in (".wav",):
                # Verify it's valid WAV and re-encode to ensure correct format
                if not _convert_to_wav(upload_path, wav_path):
                    raise HTTPException(status_code=400, detail="Invalid WAV file")
            else:
                # Convert other formats via ffmpeg
                if not _convert_to_wav(upload_path, wav_path):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to convert {suffix} to WAV. Is ffmpeg installed?",
                    )

            # Transcribe
            text, duration = dictate.transcribe_file(wav_path)

            if response_format == "text":
                return PlainTextResponse(content=text)
            elif response_format == "verbose_json":
                return JSONResponse(
                    content={
                        "task": "transcribe",
                        "language": language,
                        "duration": round(duration, 2),
                        "text": text,
                        "segments": [],
                        "words": [],
                    }
                )
            else:  # "json" (default)
                return JSONResponse(content={"text": text})

        finally:
            upload_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "model_loaded": dictate.model is not None,
            "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2)
            if torch.cuda.is_available()
            else 0,
        }

    return app


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    dictate = CanaryDictate()

    # Load model (this takes a while first time — downloads ~5GB)
    dictate.load_model()

    # Start HTTP API server in background thread
    import uvicorn

    app = create_api(dictate)
    api_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": API_HOST, "port": API_PORT, "log_level": "warning"},
        daemon=True,
    )
    api_thread.start()
    log.info(
        "HTTP API listening on http://%s:%d/v1/audio/transcriptions", API_HOST, API_PORT
    )

    # Write PID file for easy signal delivery
    pid_file = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "canary-dictate.pid"
    pid_file.write_text(str(os.getpid()))
    log.info("PID %d written to %s", os.getpid(), pid_file)

    # Signal handlers
    def on_toggle(signum, frame):
        log.debug("Received SIGUSR1 (toggle)")
        # Run in thread to avoid signal handler limitations
        threading.Thread(target=dictate.toggle, daemon=True).start()

    def on_cancel(signum, frame):
        log.debug("Received SIGUSR2 (cancel)")
        threading.Thread(target=dictate.cancel_recording, daemon=True).start()

    def on_shutdown(signum, frame):
        log.info("Shutting down...")
        dictate.cancel_recording()
        pid_file.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGUSR1, on_toggle)
    signal.signal(signal.SIGUSR2, on_cancel)
    signal.signal(signal.SIGTERM, on_shutdown)
    signal.signal(signal.SIGINT, on_shutdown)

    log.info("Canary Dictate daemon ready. Send SIGUSR1 to toggle, SIGUSR2 to cancel.")

    # Keep main thread alive
    try:
        while True:
            signal.pause()
    except KeyboardInterrupt:
        on_shutdown(None, None)


if __name__ == "__main__":
    main()
