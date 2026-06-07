import os
import io
import base64
import subprocess
import tempfile
from dataclasses import dataclass

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
DEFAULT_OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "opus").lower()
SAMPLE_RATE = 24000
SUPPORTED_LANGUAGES = {"en", "zh"}


@dataclass(frozen=True)
class AudioFormatSpec:
    name: str
    mime: str
    extension: str
    default_bitrate_kbps: int | None = None


OUTPUT_FORMATS: dict[str, AudioFormatSpec] = {
    "opus": AudioFormatSpec("opus", "audio/opus", "opus", 24),
    "mp3": AudioFormatSpec("mp3", "audio/mpeg", "mp3", 64),
    "wav": AudioFormatSpec("wav", "audio/wav", "wav"),
}

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


def _normalize_output_format(value: str) -> str:
    fmt = value.strip().lower()
    if fmt in {"ogg", "oga"}:
        return "opus"
    return fmt


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


def _tensor_to_waveform(audio_tensor: torch.Tensor) -> np.ndarray:
    """Convert model output (often bfloat16 on CUDA) to mono float32 numpy."""
    waveform = audio_tensor.detach().float().cpu().numpy().squeeze()
    if waveform.ndim > 1:
        waveform = waveform.reshape(-1)
    return np.clip(waveform, -1.0, 1.0).astype(np.float32)


def _encode_wav(waveform: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _encode_ffmpeg(waveform: np.ndarray, *, codec_args: list[str]) -> bytes:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        *codec_args,
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        input=waveform.tobytes(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Audio encoding failed: {err or 'ffmpeg error'}")
    if not proc.stdout:
        raise RuntimeError("Audio encoding failed: ffmpeg returned empty output.")
    return proc.stdout


def _encode_opus(waveform: np.ndarray, bitrate_kbps: int) -> bytes:
    return _encode_ffmpeg(
        waveform,
        codec_args=[
            "-c:a",
            "libopus",
            "-b:a",
            f"{bitrate_kbps}k",
            "-application",
            "voip",
            "-vbr",
            "on",
            "-compression_level",
            "10",
            "-f",
            "opus",
        ],
    )


def _encode_mp3(waveform: np.ndarray, bitrate_kbps: int) -> bytes:
    return _encode_ffmpeg(
        waveform,
        codec_args=[
            "-c:a",
            "libmp3lame",
            "-b:a",
            f"{bitrate_kbps}k",
            "-f",
            "mp3",
        ],
    )


def _encode_audio(
    waveform: np.ndarray,
    output_format: str,
    *,
    bitrate_kbps: int | None,
) -> tuple[bytes, AudioFormatSpec]:
    spec = OUTPUT_FORMATS[output_format]
    if output_format == "wav":
        return _encode_wav(waveform), spec
    if output_format == "opus":
        return _encode_opus(waveform, bitrate_kbps or spec.default_bitrate_kbps or 24), spec
    if output_format == "mp3":
        return _encode_mp3(waveform, bitrate_kbps or spec.default_bitrate_kbps or 64), spec
    raise ValueError(f"Unsupported output format '{output_format}'.")


def synthesize_speech(
    text: str,
    voice_sample: np.ndarray,
    *,
    speaker_label: str,
    cfg_scale: float,
    ddpm_steps: int,
    output_format: str,
    output_bitrate_kbps: int | None,
) -> dict:
    """Generate cloned speech and return encoded audio metadata."""
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

    waveform = _tensor_to_waveform(outputs.speech_outputs[0])
    duration_seconds = round(len(waveform) / SAMPLE_RATE, 3)
    audio_bytes, spec = _encode_audio(
        waveform,
        output_format,
        bitrate_kbps=output_bitrate_kbps,
    )

    return {
        "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
        "audio_mime": spec.mime,
        "format": spec.name,
        "extension": spec.extension,
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": duration_seconds,
        "audio_bytes": len(audio_bytes),
        "bitrate_kbps": output_bitrate_kbps or spec.default_bitrate_kbps,
    }


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
        "ddpm_steps": int,            # optional, default 5
        "output_format": str,         # optional, default "opus" ("opus", "mp3", "wav")
        "output_bitrate_kbps": int    # optional, opus default 24, mp3 default 64
      }
    }

    Returns:
    {
      "audio_base64": str,
      "audio_mime": str,
      "format": str,
      "extension": str,
      "sample_rate": 24000,
      "duration_seconds": float,
      "audio_bytes": int,
      "bitrate_kbps": int | null,
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
    output_format = _normalize_output_format(
        str(job_input.get("output_format", DEFAULT_OUTPUT_FORMAT))
    )
    output_bitrate_raw = job_input.get("output_bitrate_kbps")
    output_bitrate_kbps = (
        int(output_bitrate_raw) if output_bitrate_raw is not None else None
    )

    if not isinstance(text, str) or not text.strip():
        return {"error": "Missing required 'text' (non-empty string)."}

    print(
        f"[VibeVoice] Job received: text_chars={len(text.strip())}, "
        f"has_audio_b64={isinstance(audio_b64, str) and bool(audio_b64.strip())}, "
        f"language={language}, output_format={output_format}, device={device}"
    )

    if not isinstance(audio_b64, str) or not audio_b64.strip():
        return {"error": "Missing required 'audio_b64' (base64-encoded reference audio)."}

    if language not in SUPPORTED_LANGUAGES:
        return {"error": f"Unsupported language '{language}'. Use 'en' or 'zh'."}

    if output_format not in OUTPUT_FORMATS:
        supported = ", ".join(sorted(OUTPUT_FORMATS))
        return {
            "error": f"Unsupported output_format '{output_format}'. Use one of: {supported}."
        }

    if cfg_scale <= 0:
        return {"error": "'cfg_scale' must be a positive number."}

    if ddpm_steps < 1:
        return {"error": "'ddpm_steps' must be an integer >= 1."}

    if output_bitrate_kbps is not None and output_bitrate_kbps < 8:
        return {"error": "'output_bitrate_kbps' must be >= 8 when provided."}

    try:
        voice_sample = _decode_voice_sample(audio_b64.strip())
        print("[VibeVoice] Starting inference...")
        result = synthesize_speech(
            text.strip(),
            voice_sample,
            speaker_label=str(speaker_label),
            cfg_scale=cfg_scale,
            ddpm_steps=ddpm_steps,
            output_format=output_format,
            output_bitrate_kbps=output_bitrate_kbps,
        )
        print(
            f"[VibeVoice] Inference complete: format={result['format']}, "
            f"duration={result['duration_seconds']}s, bytes={result['audio_bytes']}"
        )
        return {
            **result,
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
