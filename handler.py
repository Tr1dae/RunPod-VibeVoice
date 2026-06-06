import os
import io
import base64
import tempfile

import librosa
import numpy as np
import runpod
import soundfile as sf
import torch

from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)

# --- Env ---
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/VibeVoice-Large")
DEFAULT_LANGUAGE = os.getenv("LANGUAGE", "en")
DEFAULT_SPEAKER_LABEL = os.getenv("SPEAKER_LABEL", "Speaker 0")
DEFAULT_CFG_SCALE = float(os.getenv("CFG_SCALE", "1.3"))
DEFAULT_DDPM_STEPS = int(os.getenv("DDPM_STEPS", "5"))
SAMPLE_RATE = 24000
SUPPORTED_LANGUAGES = {"en", "zh"}

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

# --- Load once at startup ---
print(
    f"[VibeVoice] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, "
    f"cuda_devices={cuda_count}"
)
if device == "cpu":
    print(
        "[VibeVoice] WARNING: running on CPU — inference on VibeVoice-Large will be extremely slow. "
        "Rebuild the image with CUDA-enabled PyTorch (see Dockerfile)."
    )
print(f"[VibeVoice] Loading model from '{MODEL_PATH}' on {device}...")
processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
).to(device).eval()
model.set_ddpm_inference_steps(DEFAULT_DDPM_STEPS)
print("[VibeVoice] Model ready.")


def _decode_voice_sample(audio_b64: str) -> np.ndarray:
    """Decode base64 audio bytes and return 24 kHz mono float32 waveform."""
    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 in 'audio_b64'.") from exc

    if not audio_bytes:
        raise ValueError("'audio_b64' decoded to empty bytes.")

    # librosa + ffmpeg handles wav/mp3/flac/ogg reference clips
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        voice, sr = librosa.load(tmp.name, sr=None, mono=True)

    if voice.size == 0:
        raise ValueError("Reference audio contains no samples.")

    if sr != SAMPLE_RATE:
        voice = librosa.resample(voice, orig_sr=sr, target_sr=SAMPLE_RATE)

    return voice.astype(np.float32)


def _format_prompt(text: str, speaker_label: str) -> str:
    """Ensure the prompt includes a speaker prefix for single-speaker cloning."""
    stripped = text.strip()
    if ":" in stripped.split("\n", 1)[0]:
        return stripped
    return f"{speaker_label}: {stripped}"


def _audio_to_base64_wav(audio_tensor: torch.Tensor) -> str:
    """Convert model output tensor to base64-encoded WAV at 24 kHz."""
    waveform = audio_tensor.detach().cpu().numpy().squeeze().astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def synthesize_speech(
    text: str,
    voice_sample: np.ndarray,
    *,
    speaker_label: str,
    cfg_scale: float,
    ddpm_steps: int,
) -> str:
    """Generate cloned speech and return base64-encoded WAV."""
    model.set_ddpm_inference_steps(ddpm_steps)

    prompt = _format_prompt(text, speaker_label)
    inputs = processor(
        text=[prompt],
        voice_samples=[[voice_sample]],
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
        )

    return _audio_to_base64_wav(outputs.speech_outputs[0])


def handler(job):
    """
    Expects:
    {
      "input": {
        "text": str,                  # required
        "audio_b64": str,             # required - base64 reference clip for cloning
        "audio_mime": str,            # optional, informational only
        "speaker_label": str,         # optional, default "Speaker 0"
        "language": "en"|"zh",        # optional, default "en"
        "cfg_scale": float,           # optional, default 1.3
        "ddpm_steps": int             # optional, default 5
      }
    }

    Returns:
    {
      "audio_base64": str,
      "sample_rate": 24000,
      "language": str,
      "speaker_label": str,
      "cfg_scale": float,
      "ddpm_steps": int
    }
    """
    job_input = job.get("input", {})

    text = job_input.get("text")
    audio_b64 = job_input.get("audio_b64")
    speaker_label = job_input.get("speaker_label", DEFAULT_SPEAKER_LABEL)
    language = job_input.get("language", DEFAULT_LANGUAGE)
    cfg_scale = float(job_input.get("cfg_scale", DEFAULT_CFG_SCALE))
    ddpm_steps = int(job_input.get("ddpm_steps", DEFAULT_DDPM_STEPS))

    if not isinstance(text, str) or not text.strip():
        return {"error": "Missing required 'text' (non-empty string)."}

    print(
        f"[VibeVoice] Job received: text_chars={len(text.strip())}, "
        f"has_audio_b64={isinstance(audio_b64, str) and bool(audio_b64.strip())}, "
        f"language={language}, device={device}"
    )

    if not isinstance(audio_b64, str) or not audio_b64.strip():
        return {"error": "Missing required 'audio_b64' (base64-encoded reference audio)."}

    if language not in SUPPORTED_LANGUAGES:
        return {"error": f"Unsupported language '{language}'. Use 'en' or 'zh'."}

    if cfg_scale <= 0:
        return {"error": "'cfg_scale' must be a positive number."}

    if ddpm_steps < 1:
        return {"error": "'ddpm_steps' must be an integer >= 1."}

    try:
        voice_sample = _decode_voice_sample(audio_b64.strip())
        print("[VibeVoice] Starting inference...")
        audio_out_b64 = synthesize_speech(
            text.strip(),
            voice_sample,
            speaker_label=str(speaker_label),
            cfg_scale=cfg_scale,
            ddpm_steps=ddpm_steps,
        )
        print("[VibeVoice] Inference complete.")
        return {
            "audio_base64": audio_out_b64,
            "sample_rate": SAMPLE_RATE,
            "language": language,
            "speaker_label": speaker_label,
            "cfg_scale": cfg_scale,
            "ddpm_steps": ddpm_steps,
        }
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc


runpod.serverless.start({"handler": handler})
