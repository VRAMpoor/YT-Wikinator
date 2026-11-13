import sys
import json
import os
from pathlib import Path

# Add DLL search directories for CUDA / cuBLAS on Windows
if sys.platform == "win32":
    # All relative to this script / this Python, so the project can be moved.
    # ytc_setup.py reads this list (`candidates`) for its GPU check.
    here = Path(__file__).resolve().parent
    candidates = [
        here,
        here / "bin" / "llama",
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        # faster-whisper on CUDA needs cuDNN as well (pip package nvidia-cudnn-cu12).
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
    ]
    for c in candidates:
        if c.exists():
            try:
                os.add_dll_directory(str(c))
                os.environ["PATH"] = str(c) + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: transcribe_cuda.py <audio_file> [model_size]\n")
        sys.exit(1)

    audio_path = sys.argv[1]
    # The multilingual model: it detects the spoken language itself.
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"

    try:
        from faster_whisper import WhisperModel
        use_cuda = True
        try:
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
        except Exception:
            use_cuda = False
            model = WhisperModel(model_size, device="cpu", compute_type="int8")

        cues = []
        try:
            segments, info = model.transcribe(audio_path, beam_size=1)
            for s in segments:
                text = s.text.strip()
                if text:
                    cues.append({
                        "start_sec": round(float(s.start), 3),
                        "end_sec": round(float(s.end), 3),
                        "text": text,
                    })
        except Exception as exec_err:
            if use_cuda:
                sys.stderr.write(f"[Whisper] CUDA execution note ({exec_err}). Retrying with high-speed CPU int8.\n")
                use_cuda = False
                cues = []  # drop cues from the aborted CUDA pass, the CPU pass starts over
                cpu_model = WhisperModel(model_size, device="cpu", compute_type="int8")
                segments, info = cpu_model.transcribe(audio_path, beam_size=1)
                for s in segments:
                    text = s.text.strip()
                    if text:
                        cues.append({
                            "start_sec": round(float(s.start), 3),
                            "end_sec": round(float(s.end), 3),
                            "text": text,
                        })
            else:
                raise exec_err

        # stderr only: stdout must stay pure JSON for ytc.
        device = "CUDA GPU (float16)" if use_cuda else "CPU (int8)"
        language = getattr(info, "language", None) or "unknown"
        sys.stderr.write(f"[Whisper] Transcribed on {device}, model {model_size}, language {language}.\n")
        print(json.dumps(cues))
    except Exception as e:
        # Exit non-zero so ytc shows this reason instead of "empty transcript".
        sys.stderr.write(f"Whisper transcription error: {type(e).__name__}: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
