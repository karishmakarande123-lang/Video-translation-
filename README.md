# s2st_en_mr — baseline setup & run

The **baseline** ASR→MT→TTS pipeline and a local Flask **GUI** are
implemented (prosody, isochrony, speaker, and eval modules are planned but
not built yet).

## 1. Environment

```bash
cd s2st_en_mr
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux

pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`torch` and `torchaudio` **must** be the same version from the same
CUDA-build index — installing them separately/later (e.g. `pip install
torchaudio` on its own) grabs whatever's newest and silently mismatches,
which fails at import time on Windows with
`OSError: [WinError 127] The specified procedure could not be found`
(confirmed empirically on this project).

Models auto-download from Hugging Face on first use of each stage. To
pre-fetch all of them up front (recommended — shows progress bars and
surfaces network/auth problems before your first real run) into
`.cache/s2st_en_mr/` (set by `configs/hardware.yaml`):

```bash
python -m scripts.download_models
```

- faster-whisper `small` (~500MB)
- `ai4bharat/indictrans2-en-indic-dist-200M` (~800MB)
- `ai4bharat/IndicF5` (~1-2GB)

`ai4bharat/indictrans2-en-indic-dist-200M` and `ai4bharat/IndicF5` are
**gated** repos (confirmed in practice, despite being publicly listed):

1. Log into huggingface.co in your browser, visit each model page, and
   click "Agree/Request access":
   - https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M
   - https://huggingface.co/ai4bharat/IndicF5 — this one may require the
     ai4bharat maintainers to manually approve your request rather than
     granting instantly; if `download_models.py` fails with "you are not
     in the authorized list", you're waiting on that approval.
2. Create a read-scoped token: https://huggingface.co/settings/tokens
3. `hf auth login` (the `huggingface-cli login` alias also works but is
   deprecated) and paste the token.

Never paste a token into a chat/AI assistant — if you ever do, revoke and
regenerate it afterward.
## 2. Run the baseline from the CLI

```bash
python -m scripts.run_baseline ^
    --audio path\to\english_input.wav ^
    --ref-audio path\to\marathi_reference_clip.wav ^
    --ref-text "मराठी संदर्भ मजकूर" ^
    --verbose
```

- `--audio`: English speech to translate.
- `--ref-audio`: a few seconds of Marathi speech from the target voice
  (used by IndicF5 for voice cloning).
- `--ref-text`: exact transcript of `--ref-audio`. Optional but strongly
  recommended — if omitted, the TTS stage auto-transcribes the reference
  clip on CPU as a fallback, which is lower fidelity.

Output: transcribed English text, translated Marathi text, and a
synthesized `.wav` path (printed at the end, file left under
`.cache/s2st_en_mr/`).

## 3. Run the GUI

```bash
python -m gui.app
```

Open http://127.0.0.1:7860 in a browser. Upload English audio (or record it
in-browser via the "Record" button, which needs mic permission), upload a
Marathi reference clip + its transcript, and click **Run Pipeline**. Stage
progress (ASR → MT → TTS) streams live; the synthesized audio plays inline
when done.

- Only one pipeline run is allowed at a time (enforced in `gui/state.py`) —
  the 6GB card can't run two overlapping requests.
- The GUI never preloads models at startup; the first request after
  launching the server will be slower while models load into cache/VRAM.
- Server host/port come from `configs/hardware.yaml` -> `gui:`.

## 4. Hardware notes (RTX 3060 Laptop, 6GB VRAM)

- Stages load sequentially and unload+`empty_cache()` between each other —
  never two models resident on GPU at once. Don't change
  `sequential_stage_loading` in `configs/hardware.yaml`.
- Stick to faster-whisper `small`/`medium`; `large-v3` needs a bigger GPU
  (Kaggle P100 16GB or a college server).
- If you hit CUDA OOM on the TTS stage, drop `tts.dtype` to `"fp16"` (already
  default) and confirm nothing else is holding VRAM (`nvidia-smi`).
- `transformers` is pinned to exactly `4.44.2` in `requirements.txt` — don't
  bump it casually. Newer versions break IndicTrans2's custom
  `trust_remote_code` model code in two different ways (see the comment in
  `requirements.txt`); confirmed empirically on this project.
- `torch`/`torchaudio` must be installed together, same version, same CUDA
  index — see step 1. IndicF5's `f5-tts` dependency needs `torchaudio`.
- IndicF5's official HF loading path has two upstream bugs, both worked
  around in this project (not something you need to redo): a
  `load_model() missing ckpt_path` crash (fixed via ai4bharat's own GitHub
  package instead of PyPI `f5-tts`), and a silent weight-loading failure
  for ~16 parameters that otherwise makes the model output noise instead
  of speech (fixed in `baseline/tts.py`'s `_patch_undertrained_weights`).
  See the comments in `requirements.txt` and `baseline/tts.py` for details.
  Verified by feeding synthesized output back through faster-whisper and
  confirming it transcribes back to something resembling the input text.
- A third IndicF5 issue, separate from the two above: its raw output
  routinely exceeds +/-1.0 (peaks measured around 1.2-1.4). `sf.write()`
  defaults to 16-bit PCM for `.wav`, which hard-clips (not scales) anything
  outside that range — measured 94-97% of samples clipped, which is what
  "very noisy" output actually was, not a model-quality problem.
  `baseline/tts.py`'s `synthesize()` peak-normalizes only when the signal
  exceeds +/-1.0, leaving headroom at 0.98. Confirmed fixed: 0% clipping
  and a normal speech RMS (~0.13 vs the previous ~0.98) after the fix.

## 5. Not implemented yet

`prosody/`, `isochrony/`, `speaker/`, `eval/`, and the other `scripts/`
entries beyond `run_baseline.py`/`download_models.py` are directory stubs
only (`__init__.py`). The `mode="baseline+research"` path in
`baseline/pipeline.py` deliberately raises `NotImplementedError` until
those modules exist — the GUI's second mode option is disabled accordingly.
The GUI's metrics panel will stay empty until `eval/` is built.
