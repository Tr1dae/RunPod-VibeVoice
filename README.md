# RunPod VibeVoice — Voice-Cloning TTS Worker

RunPod serverless GPU worker for [aoi-ot/VibeVoice-Large](https://huggingface.co/aoi-ot/VibeVoice-Large). Send a text prompt plus a reference audio clip; the worker clones that voice and returns synthesized speech at **24 kHz**.

**Repository:** [Tr1dae/RunPod-VibeVoice](https://github.com/Tr1dae/RunPod-VibeVoice)  
**Docker image:** `docker.io/qualitycontrolty/runpod-vibevoice:latest`

## What it does

- Loads **VibeVoice-Large** and the **Qwen2.5-7B** tokenizer from disk (baked into the image at build time).
- Accepts a **base64-encoded reference audio** file (`audio_b64`) for voice cloning.
- Returns **base64 WAV** output at 24 kHz.
- Supports **English** and **Chinese** (`language`: `en` or `zh`).
- Exposes inference tuning via `cfg_scale` and `ddpm_steps`.

Inference code comes from the community-maintained fork: [vibevoice-community/VibeVoice](https://github.com/vibevoice-community/VibeVoice).

## Hardware (RunPod)

| Setting | Recommendation |
|--------|----------------|
| GPU | NVIDIA with **≥24 GB VRAM** (e.g. RTX 4090, A5000) |
| Container disk | **≥40 GB** (model + tokenizer are embedded in the image) |
| CUDA | 12.x |

Cold start loads the full model from local paths; no Hugging Face download at runtime (`HF_HUB_OFFLINE=1`).

## RunPod setup

1. Create a **Serverless** endpoint.
2. Set the container image to:
   ```
   docker.io/qualitycontrolty/runpod-vibevoice:latest
   ```
3. Optional environment variables (defaults shown):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/VibeVoice-Large` | Local path to baked-in weights |
| `CFG_SCALE` | `1.3` | Default classifier-free guidance (override per job) |
| `DDPM_STEPS` | `5` | Default diffusion steps (override per job) |
| `SPEAKER_LABEL` | `Speaker 0` | Default speaker prefix in the prompt |
| `LANGUAGE` | `en` | Default language hint if omitted in job input |

See [`.runpod/hub.json`](.runpod/hub.json) for RunPod Hub metadata and [`.runpod/tests.json`](.runpod/tests.json) for example test jobs (includes a sample `audio_b64` clip).

## API — request

POST a job with this shape:

```json
{
  "input": {
    "text": "Hello world, this is VibeVoice speaking.",
    "audio_b64": "<base64-encoded reference audio (WAV, MP3, etc.)>",
    "language": "en",
    "speaker_label": "Speaker 0",
    "cfg_scale": 1.3,
    "ddpm_steps": 5
  }
}
```

### Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `text` | Yes | — | Text to synthesize. If it has no `Speaker N:` prefix, `speaker_label` is prepended automatically. |
| `audio_b64` | Yes | — | Base64-encoded reference clip used for voice cloning. Decoded with ffmpeg/librosa (mono, resampled to 24 kHz). |
| `language` | No | `en` | `en` or `zh` only. |
| `speaker_label` | No | `Speaker 0` | Speaker prefix for single-speaker prompts. |
| `cfg_scale` | No | `1.3` | CFG strength. Higher can sound more literal but less natural. |
| `ddpm_steps` | No | `5` | Diffusion denoising steps. More steps = slower, often higher quality. |
| `audio_mime` | No | — | Accepted for documentation; not used by the handler today. |

**Note:** Requests with only `text` and `language` (no `audio_b64`) return an error. Reference audio is required for this deployment.

## API — response

### Success

```json
{
  "audio_base64": "<base64 WAV at 24 kHz>",
  "sample_rate": 24000,
  "language": "en",
  "speaker_label": "Speaker 0",
  "cfg_scale": 1.3,
  "ddpm_steps": 5
}
```

Decode `audio_base64` to bytes and save as a `.wav` file.

### Validation error (job completes, error in body)

```json
{
  "error": "Missing required 'audio_b64' (base64-encoded reference audio)."
}
```

### Inference failure

The handler raises `RuntimeError`; RunPod marks the job as **FAILED**.

## Encoding reference audio

Example in Python:

```python
import base64

with open("reference.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "input": {
        "text": "Hello world, this is VibeVoice speaking.",
        "audio_b64": audio_b64,
        "language": "en",
        "cfg_scale": 1.3,
        "ddpm_steps": 5,
    }
}
```

For a minimal test clip without your own file, copy the `audio_b64` value from [`.runpod/tests.json`](.runpod/tests.json).

## Build the image locally

Requires Docker, ~80+ GB free disk for build layers, and network access for the first build (downloads models).

```bash
docker build --platform linux/amd64 -t qualitycontrolty/runpod-vibevoice:latest .
```

The image embeds:

- `aoi-ot/VibeVoice-Large` → `/app/models/VibeVoice-Large`
- `Qwen/Qwen2.5-7B` (tokenizer) → `/app/models/Qwen2.5-7B`

Subsequent rebuilds after changing only `handler.py` reuse cached model layers and are much faster.

Push to Docker Hub:

```bash
docker push qualitycontrolty/runpod-vibevoice:latest
```

## Project layout

| Path | Purpose |
|------|---------|
| `handler.py` | RunPod serverless entrypoint |
| `Dockerfile` | Full image build (code + models) |
| `requirements.txt` | Extra pip deps (`runpod`, `librosa`, etc.) |
| `.runpod/hub.json` | RunPod Hub template config |
| `.runpod/tests.json` | Example serverless test payloads |

## Responsible use

VibeVoice is intended for **research**. High-quality synthetic speech can be misused. Use only with consent for voice cloning, comply with applicable laws, and disclose AI-generated audio when sharing outputs. See the [model card](https://huggingface.co/aoi-ot/VibeVoice-Large) for limitations (English/Chinese, speech-only, no overlapping speech, etc.).

## Credits

- [Microsoft VibeVoice](https://microsoft.github.io/VibeVoice) — original research
- [vibevoice-community/VibeVoice](https://github.com/vibevoice-community/VibeVoice) — inference code used in this worker
- [aoi-ot/VibeVoice-Large](https://huggingface.co/aoi-ot/VibeVoice-Large) — model weights
