import sys
import json
import os
from pathlib import Path

# The NVIDIA libraries faster-whisper (ctranslate2 4.x) needs on the GPU on
# Windows: cuBLAS 12 and the cuDNN 9 sub-libraries. ctranslate2 ships only the
# cuDNN dispatcher (cudnn64_9.dll); when a sub-library is missing it ends the
# whole process at the first convolution, so no except clause below and no CPU
# retry would run. main() therefore checks them before it tries CUDA.
CUDA_LIBRARIES = ("cublas64_12.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll")

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
    # pip puts the packages into the per-user site when it cannot write to
    # this Python's own folder.
    try:
        import site
        user_site = Path(site.getusersitepackages())
        candidates += [
            user_site / "nvidia" / "cublas" / "bin",
            user_site / "nvidia" / "cudnn" / "bin",
        ]
    except Exception:
        pass
    for c in candidates:
        if c.exists():
            try:
                os.add_dll_directory(str(c))
                os.environ["PATH"] = str(c) + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass


def missing_cuda_libraries(folders, load=None):
    """Which of CUDA_LIBRARIES cannot be loaded. Each is tried by name (the
    folders added with os.add_dll_directory) and then by its full path in one
    of `folders` (PATH, where the cuDNN dispatcher also looks). A library that
    loads stays loaded, so the dispatcher later gets that same copy."""
    if load is None:
        import ctypes
        load = ctypes.WinDLL
    missing = []
    for name in CUDA_LIBRARIES:
        tries = [name]
        for folder in folders:
            try:
                if folder and (Path(folder) / name).is_file():
                    tries.append(str(Path(folder) / name))
            except (OSError, ValueError):
                continue
        for target in tries:
            try:
                load(target)
                break
            except OSError:
                continue
        else:
            missing.append(name)
    return missing


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
        missing = []
        if sys.platform == "win32":
            missing = missing_cuda_libraries(os.environ.get("PATH", "").split(os.pathsep))
        if missing:
            use_cuda = False
            sys.stderr.write(f"[Whisper] GPU libraries not found ({', '.join(missing)}); using the CPU.\n")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        else:
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


def self_test() -> int:
    """Checks without a model or GPU:  python transcribe_cuda.py --self-test"""
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def test_all_libraries_load(self):
            loaded = []
            self.assertEqual(missing_cuda_libraries([], load=loaded.append), [])
            self.assertEqual(loaded, list(CUDA_LIBRARIES))

        def test_missing_cudnn_means_cpu(self):
            def load(target):
                if "cudnn" in target:
                    raise OSError("not found")
            self.assertEqual(missing_cuda_libraries(["", "nowhere"], load=load),
                             ["cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"])

        def test_full_path_in_a_folder_is_tried(self):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "cudnn_ops64_9.dll").write_bytes(b"x")
                loaded = []

                def load(target):
                    if not Path(target).is_absolute():
                        raise OSError("not on the DLL search path")
                    loaded.append(target)
                missing = missing_cuda_libraries([tmp], load=load)
                self.assertEqual(missing, ["cublas64_12.dll", "cudnn_cnn64_9.dll"])
                self.assertEqual(loaded, [str(Path(tmp) / "cudnn_ops64_9.dll")])

        @unittest.skipUnless(sys.platform == "win32", "Windows only")
        def test_user_site_is_searched(self):
            import site
            user_site = Path(site.getusersitepackages())
            self.assertIn(user_site / "nvidia" / "cudnn" / "bin", candidates)
            self.assertIn(user_site / "nvidia" / "cublas" / "bin", candidates)

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    main()
