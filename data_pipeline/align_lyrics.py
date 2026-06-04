import json
import torch
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import librosa
import difflib

# Load true lyrics
with open("data/raw/nus48e/lyrics.json") as f:
    true_lyrics = json.load(f)

# Load existing manifest which has audio_path, start, end
manifest_path = Path("data/nus48e_processed/manifest.json")
clips = json.loads(manifest_path.read_text())

device = "cuda" if torch.cuda.is_available() else "cpu"
proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)

def get_transcription(audio_path, start, end):
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1: audio = audio.mean(1)
    if end > start:
        start_frame, frames = int(start*sr), int((end-start)*sr)
        audio = audio[start_frame:start_frame+frames]
    
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    peak = abs(audio).max()
    if peak > 0: audio /= peak
    
    inp = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        ids = model.generate(inp, max_new_tokens=225, language="en", task="transcribe")
    text = proc.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip().lower()
    # remove punctuation
    import re
    return re.sub(r'[^\w\s]', '', text)

# We will align sequentially for each song
# Group clips by (singer, song)
from collections import defaultdict
grouped = defaultdict(list)
for c in clips:
    grouped[(c["metadata"]["singer"], c["metadata"]["song"])].append(c)

aligned_clips = []

import re
def normalize(t): return re.sub(r'[^\w\s]', '', t.lower()).split()

print("Aligning...")
for (singer, song), song_clips in grouped.items():
    song_clips.sort(key=lambda x: x["start"])
    lyrics_words = normalize(true_lyrics[song])
    
    current_word_idx = 0
    for c in song_clips:
        pred_text = get_transcription(c["audio_path"], c["start"], c["end"])
        pred_words = normalize(pred_text)
        
        if not pred_words:
            c["text"] = ""
            aligned_clips.append(c)
            continue
            
        # Search for pred_words in the remaining lyrics
        best_match_idx = current_word_idx
        best_match_score = 0
        best_match_len = len(pred_words)
        
        search_window = min(len(lyrics_words), current_word_idx + 30)
        
        # simple heuristic: match the first few words and last few words
        for i in range(current_word_idx, search_window):
            for length in range(max(1, len(pred_words)-5), min(len(lyrics_words)-i+1, len(pred_words)+8)):
                cand = lyrics_words[i:i+length]
                sm = difflib.SequenceMatcher(None, pred_words, cand)
                score = sm.ratio()
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = i
                    best_match_len = length
        
        if best_match_score > 0.3:
            matched_text = " ".join(lyrics_words[best_match_idx:best_match_idx+best_match_len])
            current_word_idx = best_match_idx + best_match_len
        else:
            # Fallback to zero-shot if we can't align at all (should be rare)
            matched_text = " ".join(pred_words)
            
        c["text"] = matched_text
        aligned_clips.append(c)
        print(f"[{singer}-{song}] PRED: {pred_text}  -->  ALIGNED: {matched_text}")

with open("data/nus48e_processed/manifest_aligned.json", "w") as f:
    json.dump(aligned_clips, f, indent=2)
print("Saved manifest_aligned.json")
