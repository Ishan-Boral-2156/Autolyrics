#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AutoLyrics - Complete End-to-End Pipeline.

Runs: data loading → baseline eval → LoRA fine-tune → eval → PDF report.
Self-contained to avoid import-chain issues in the project modules.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, time, json, math, random, re, unicodedata
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType
import jiwer

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# ── Config ────────────────────────────────────────────────────
MODEL_NAME = "openai/whisper-tiny"
OUTPUT_DIR = Path("runs/autolyrics_complete")
REPORTS_DIR = Path("reports")
SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 30  # Whisper's hard limit

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    USE_FP16 = True
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    USE_FP16 = False
else:
    DEVICE = torch.device("cpu")
    USE_FP16 = False

print(f"Device: {DEVICE} | FP16: {USE_FP16}")

# Training hyper-params
TRAIN_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
MAX_STEPS = 1000
LORA_R, LORA_ALPHA, LORA_DROPOUT = 32, 64, 0.05


# ── Data Loading ─────────────────────────────────────────────
def create_dataset():
    random.seed(42)
    manifest_path = Path("data/nus48e_processed/manifest_aligned.json")
    if not manifest_path.exists():
        print(f"Error: Could not find {manifest_path}. Run align_lyrics.py first.")
        sys.exit(1)

    all_clips = json.loads(manifest_path.read_text())

    # ── FILTER OUT BAD CLIPS ──
    clean = []
    removed = {"short_text": 0, "too_long": 0, "bad_ratio": 0, "nonsense": 0}
    for c in all_clips:
        text = c["text"].strip()
        dur = c.get("end", 0) - c.get("start", 0)
        nwords = len(text.split())

        # Skip clips with <= 2 words (alignment errors)
        if nwords <= 2:
            removed["short_text"] += 1
            continue

        # Skip clips longer than 30s (Whisper can't handle them)
        if dur > MAX_AUDIO_SEC:
            removed["too_long"] += 1
            continue

        # Skip clips with extreme audio/text ratio (> 5 s/word = likely misaligned)
        if nwords > 0 and dur / nwords > 5.0:
            removed["bad_ratio"] += 1
            continue

        # Skip nonsense text (not real English words)
        if not re.search(r'[aeiou]', text):
            removed["nonsense"] += 1
            continue

        clean.append(c)

    print(f"  Filtered: {len(all_clips)} → {len(clean)} clips")
    for reason, count in removed.items():
        if count > 0:
            print(f"    removed {count} clips ({reason})")

    # Honest random split (80/10/10)
    random.shuffle(clean)
    n = len(clean)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train": clean[:n_train],
        "val": clean[n_train:n_train+n_val],
        "test": clean[n_train+n_val:]
    }

    for name, clips in splits.items():
        print(f"  {name}: {len(clips)} segments")

    return splits


# ── Dataset / Collator ───────────────────────────────────────
class SingDS(Dataset):
    def __init__(self, clips, proc):
        self.clips, self.proc = clips, proc

    def __len__(self): return len(self.clips)

    def __getitem__(self, i):
        c = self.clips[i]
        start_s = c.get("start", 0.0)
        end_s = c.get("end", 0.0)

        info = sf.info(c["audio_path"])
        sr = info.samplerate

        if end_s > start_s:
            start_frame = int(start_s * sr)
            frames = int((end_s - start_s) * sr)
            # Cap to 30s
            max_frames = int(MAX_AUDIO_SEC * sr)
            frames = min(frames, max_frames)
            audio, _ = sf.read(c["audio_path"], start=start_frame, frames=frames,
                               dtype="float32", always_2d=True)
        else:
            audio, _ = sf.read(c["audio_path"], dtype="float32", always_2d=True)

        if audio.ndim > 1: audio = audio.mean(1)
        if sr != SAMPLE_RATE:
            import torchaudio
            audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(
                torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()

        # Normalize peak
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        inp = self.proc.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

        # Tokenize with language/task prefix so labels match generation output
        labels = self.proc.tokenizer(c["text"]).input_ids

        return {
            "input_features": inp.input_features.squeeze(0),
            "labels": labels,
            "text": c["text"],
        }


class Collator:
    def __init__(self, proc): self.proc = proc

    def __call__(self, feats):
        inp = self.proc.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
        lab = self.proc.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.proc.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        return {"input_features": inp["input_features"], "labels": labels}


# ── Metrics helpers ──────────────────────────────────────────
def norm(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = re.sub(r"<\|.*?\|>", " ", t)
    t = t.lower()
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def wer_cer(preds, refs):
    p = [norm(x) for x in preds]; r = [norm(x) for x in refs]
    pairs = [(a, b) for a, b in zip(p, r) if b]
    if not pairs: return {"wer": 1.0, "cer": 1.0}
    pp, rr = zip(*pairs)
    return {"wer": float(jiwer.wer(list(rr), list(pp))),
            "cer": float(jiwer.cer(list(rr), list(pp)))}

def evaluate(model, proc, ds, device, label=""):
    print(f"\n{'─'*50}\nEvaluating: {label}\n{'─'*50}")
    model.eval()
    preds, refs, lats = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        inp = s["input_features"].unsqueeze(0).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(
                input_features=inp,
                max_new_tokens=225,
                language="en",
                task="transcribe"
            )
        lats.append(time.perf_counter() - t0)
        dec = proc.tokenizer.batch_decode(ids, skip_special_tokens=True)
        preds.append(dec[0].strip()); refs.append(s["text"])
    m = wer_cer(preds, refs)
    avg_lat = sum(lats) / len(lats) if lats else 0
    print(f"  WER: {m['wer']*100:.2f}%  CER: {m['cer']*100:.2f}%  Latency: {avg_lat:.3f}s")
    return {**m, "avg_latency": avg_lat, "predictions": preds,
            "references": refs, "latencies": lats}


# ── VRAM info ────────────────────────────────────────────────
def vram_info():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1e9
        t = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {"allocated_gb": round(a, 2), "total_gb": round(t, 2),
                "device": torch.cuda.get_device_name(0)}
    return {"allocated_gb": 0, "total_gb": 0, "device": str(DEVICE)}


# -- Main pipeline --------------------------------------------
def main():
    print("=" * 60)
    print("  AutoLyrics -- Complete Pipeline")
    print("=" * 60)

    # 1. Data
    print("\n>> PHASE 1: Creating dataset")
    splits = create_dataset()

    # 2. Load model + processor
    print("\n>> PHASE 2: Loading Whisper model")
    proc = WhisperProcessor.from_pretrained(MODEL_NAME)
    proc.tokenizer.set_prefix_tokens(language="en", task="transcribe")

    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32)
    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []
    base_model.to(DEVICE)

    test_ds = SingDS(splits["test"], proc)
    train_ds = SingDS(splits["train"], proc)
    val_ds = SingDS(splits["val"], proc)

    # 3. Baseline evaluation
    print("\n>> PHASE 3: Baseline (zero-shot) evaluation")
    baseline_results = evaluate(base_model, proc, test_ds, DEVICE, "Zero-shot baseline")

    # 4. LoRA fine-tuning
    print("\n>> PHASE 4: LoRA fine-tuning (decoder)")
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    base_model.config.use_cache = False
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    run_dir = OUTPUT_DIR / "lora_decoder"
    run_dir.mkdir(parents=True, exist_ok=True)

    collator = Collator(proc)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collator, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    # Learning rate scheduler: linear warmup then cosine decay
    total_steps = min(MAX_STEPS, len(train_loader) * TRAIN_EPOCHS)
    warmup_steps = min(50, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_FP16)

    # Use the underlying Whisper model for forward passes
    whisper_fwd = model.base_model.model

    model.train()
    print("  Starting training...")
    t0 = time.time()
    global_step = 0
    best_val_wer = float("inf")

    for epoch in range(TRAIN_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            inp = batch["input_features"].to(DEVICE)
            lab = batch["labels"].to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_FP16):
                outputs = whisper_fwd(input_features=inp, labels=lab)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            if global_step % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"    step {global_step} | loss: {loss.item():.4f} | lr: {lr:.2e}")
            if global_step >= MAX_STEPS:
                break
        avg = epoch_loss / max(1, n_batches)
        print(f"  Epoch {epoch+1}/{TRAIN_EPOCHS} | avg loss: {avg:.4f}")

        if global_step >= MAX_STEPS:
            break

    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s ({global_step} steps)")

    # Save model
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    proc.save_pretrained(str(best_dir))
    print(f"  Model saved to {best_dir}")

    # 5. Fine-tuned evaluation
    print("\n>> PHASE 5: Fine-tuned model evaluation")
    whisper_fwd.config.use_cache = True
    model.eval()
    finetuned_results = evaluate(whisper_fwd, proc, test_ds, DEVICE, "LoRA fine-tuned")

    # 6. Benchmark
    benchmark = {
        "vram": vram_info(),
        "train_time_s": round(train_time, 2),
        "device": str(DEVICE),
    }

    # 7. Save all results
    results = {
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "timestamp": datetime.now().isoformat(),
        "baseline": baseline_results,
        "finetuned": finetuned_results,
        "benchmark": benchmark,
        "lora_config": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
    }
    results_path = OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {results_path}")

    # Summary
    bw = baseline_results["wer"] * 100
    fw = finetuned_results["wer"] * 100
    rel = ((bw - fw) / bw * 100) if bw > 0 else 0
    print("\n" + "=" * 60)
    print("  [OK] PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Baseline WER:    {bw:.2f}%")
    print(f"  LoRA WER:        {fw:.2f}%")
    print(f"  Relative Delta:  {rel:+.1f}%")
    print(f"  Model checkpoint: {best_dir}")
    print(f"  Results JSON:     {results_path}")

if __name__ == "__main__":
    main()
