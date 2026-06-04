#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AutoLyrics Real-Time Demo
Plays the audio aloud while showing true Karaoke-style synchronized transcription.
Handles long songs and syncs lyrics exactly to the audio playback.
"""

import sys, io, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import warnings
warnings.filterwarnings("ignore")

import pygame
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from peft import PeftModel

MODEL_NAME = "openai/whisper-tiny"
LORA_PATH = "runs/autolyrics_complete/lora_decoder/best"

def main():
    print("=" * 70)
    print("  🎵 AutoLyrics -- Real-Time Live Demo 🎵")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Whisper on {device.upper()}...")

    # Load Model
    proc = WhisperProcessor.from_pretrained(LORA_PATH)
    proc.tokenizer.set_prefix_tokens(language="en", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []
    
    print("[*] Injecting LoRA Singing Adapters...")
    lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    lora_model.eval()

    # Allow custom audio file via command line argument
    import sys
    if len(sys.argv) > 1:
        clips = [Path(sys.argv[1])]
        if not clips[0].exists():
            print(f"Error: Could not find file {clips[0]}")
            return
    else:
        clips = list(Path("data/raw/nus48e/PMAR/sing").glob("*.wav")) + list(Path("data/raw/nus48e/PMAR/sing").glob("*.mp3"))
        if not clips:
            print("No clips found in data/raw/nus48e/PMAR/sing!")
            return

    pygame.mixer.init()

    print("\n" + "=" * 70)
    print("  Ready. Starting Live Transcription.")
    print("=" * 70)

    # Initialize pipeline for automatic chunking (handles >30s) and timestamps (for sync)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=lora_model.base_model.model,
        tokenizer=proc.tokenizer,
        feature_extractor=proc.feature_extractor,
        chunk_length_s=30,
        device=device
    )

    for i, clip in enumerate(clips):
        print(f"\n[ Loading Track: {clip.name} ]")
        print("  ⏳ Analyzing audio and generating timestamps... (takes a few seconds for full songs)")
        
        # Load audio into numpy array to avoid ffmpeg dependency
        import soundfile as sf
        audio, sr = sf.read(str(clip), dtype="float32")
        if audio.ndim > 1: audio = audio.mean(1)
        if sr != 16000:
            import torchaudio
            audio = torchaudio.transforms.Resample(sr, 16000)(
                torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()
        
        # Run pipeline on the numpy array to get chunks with stable sentence-level timestamps
        # (Word-level timestamps are buggy with chunk_length_s=30 in some HF versions)
        result = pipe(audio, return_timestamps=True)
        chunks = result.get("chunks", [])
        
        print("  ▶ Playing audio with Karaoke-style sync...\n")
        pygame.mixer.music.load(str(clip))
        pygame.mixer.music.play()
        
        for chunk in chunks:
            ts = chunk["timestamp"]
            sentence = chunk["text"].strip()
            if not sentence:
                continue
                
            start_time = ts[0] if ts[0] is not None else 0.0
            end_time = ts[1] if ts[1] is not None else start_time + 2.0
            
            # Wait for the audio to reach the exact start time of this sentence
            while True:
                pos = pygame.mixer.music.get_pos() / 1000.0
                if pos >= start_time or not pygame.mixer.music.get_busy():
                    break
                time.sleep(0.01)
                
            # Interpolate word-level typing to simulate word-level timestamps safely!
            words = sentence.split()
            if words:
                delay_per_word = (end_time - start_time) / len(words)
                sys.stdout.write("  🎙️ ")
                for word in words:
                    sys.stdout.write(word + " ")
                    sys.stdout.flush()
                    # Wait for the duration of this word
                    time.sleep(max(0.05, delay_per_word * 0.9)) # 0.9 so it finishes slightly early
                sys.stdout.write("\n")
                sys.stdout.flush()
            
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
        print("  ✅ Done.\n")
        time.sleep(1)

if __name__ == "__main__":
    main()
