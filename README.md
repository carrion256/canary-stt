# canary-stt

GPU-accelerated speech-to-text daemon using [NVIDIA Canary Qwen 2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) — a state-of-the-art English ASR model (5.63% WER on LibriSpeech). Runs the model hot in VRAM as a systemd service with two interfaces:

1. **Local dictation** — keyboard shortcut triggers recording, auto-stops on silence, types transcription into the active window via `wtype`
2. **HTTP API** — OpenAI-compatible `/v1/audio/transcriptions` endpoint for remote transcription from any device on your network

## Requirements

- NVIDIA GPU with ~5 GB VRAM (runs in bf16)
- Arch Linux / CachyOS (or any systemd-based distro)
- Hyprland / Wayland (for local dictation via `wtype`)
- Python 3.12, [uv](https://github.com/astral-sh/uv)
- `ffmpeg` (audio format conversion for the API)
- `wtype` (Wayland text input)

## Setup

```bash
# Clone
git clone <repo-url> ~/projects/canary-stt
cd ~/projects/canary-stt

# Create venv and install dependencies
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python \
  "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git" \
  "lhotse @ git+https://github.com/lhotse-speech/lhotse.git" \
  sacrebleu sounddevice fastapi uvicorn python-multipart

# Install systemd service + keybind scripts
./install.sh
```

`install.sh` symlinks everything into place:

| Source | Target |
|---|---|
| `bin/canary-dictate` | `~/.local/bin/canary-dictate` |
| `bin/canary-dictate-cancel` | `~/.local/bin/canary-dictate-cancel` |
| `systemd/canary-dictate.service` | `~/.config/systemd/user/canary-dictate.service` |

## Usage

### Start the daemon

```bash
systemctl --user start canary-dictate

# Check status (model takes ~18s to load into VRAM)
systemctl --user status canary-dictate
```

### Local dictation (Hyprland keybinds)

Add to `~/.config/hypr/hyprland.conf`:

```
bind = $mainMod, D, exec, canary-dictate          # Toggle recording
bind = $mainMod SHIFT, D, exec, canary-dictate-cancel  # Cancel recording
```

- **SUPER+D** starts recording. Speak naturally — recording auto-stops after 4 seconds of silence.
- **SUPER+D** again stops recording early and transcribes.
- **SUPER+SHIFT+D** cancels without transcribing.
- Transcribed text is typed into the active window via `wtype`.

### HTTP API

The daemon exposes an OpenAI-compatible transcription API on port **8393**.

```bash
# Basic transcription
curl -X POST http://localhost:8393/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=canary-qwen-2.5b

# {"text": "your transcription here"}
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio (OpenAI-compatible) |
| `GET` | `/v1/models` | List available models |
| `GET` | `/health` | Service health + VRAM usage |

**Transcription parameters** (multipart form):

| Field | Required | Default | Description |
|---|---|---|---|
| `file` | yes | — | Audio file (wav, mp3, m4a, flac, ogg, webm, mp4) |
| `model` | no | `canary-qwen-2.5b` | Model name (accepted but ignored) |
| `language` | no | `en` | Language hint |
| `response_format` | no | `json` | `json`, `text`, or `verbose_json` |
| `temperature` | no | `0.0` | Accepted for compatibility (not used) |

**Response formats:**

```bash
# json (default)
{"text": "Hello world"}

# text
Hello world

# verbose_json
{"task": "transcribe", "language": "en", "duration": 3.14, "text": "Hello world", "segments": [], "words": []}
```

### Remote transcription

Point any OpenAI-compatible client at your machine over Tailscale/Netbird/VPN:

```python
from openai import OpenAI

client = OpenAI(base_url="http://<tailscale-ip>:8393/v1", api_key="unused")
with open("recording.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(model="canary-qwen-2.5b", file=f)
    print(transcript.text)
```

## Performance

Measured on RTX 5090 (Blackwell, 32 GB VRAM) with bf16 inference:

| Metric | Value |
|---|---|
| Model load time | ~18s |
| VRAM usage | 5.14 GB |
| Transcription speed | 10-28x realtime |
| Latency (10s audio) | ~0.4-0.9s |

## Configuration

Edit constants at the top of `canary_dictate.py`:

```python
SILENCE_TIMEOUT = 4.0    # Seconds of silence before auto-stop
SILENCE_THRESHOLD = 0.01 # RMS threshold for silence detection
MAX_RECORD_SECS = 600    # Maximum recording length (10 min)
API_HOST = "0.0.0.0"     # API listen address
API_PORT = 8393           # API port
```

## Project structure

```
canary-stt/
├── canary_dictate.py              # Main daemon (model, recording, API)
├── install.sh                     # Symlink installer
├── bin/
│   ├── canary-dictate             # Toggle recording (sends SIGUSR1)
│   └── canary-dictate-cancel      # Cancel recording (sends SIGUSR2)
├── systemd/
│   └── canary-dictate.service     # Systemd user service
├── .venv/                         # Python 3.12 venv (not committed)
└── .gitignore
```

## License

MIT
