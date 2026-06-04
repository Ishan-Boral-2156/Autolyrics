import os
import json
import urllib.request
from pathlib import Path

CMUDICT_URL = "https://raw.githubusercontent.com/Alexir/CMUdict/master/cmudict-0.7b"
CACHE_DIR = Path(".cache")

def download_cmudict():
    CACHE_DIR.mkdir(exist_ok=True)
    dict_path = CACHE_DIR / "cmudict.txt"
    if not dict_path.exists():
        print("Downloading CMU dict...")
        urllib.request.urlretrieve(CMUDICT_URL, dict_path)
    return dict_path

def build_reverse_dict(dict_path):
    import re
    rev_dict = {}
    with open(dict_path, "r", encoding="latin1") as f:
        for line in f:
            if line.startswith(";;;"): continue
            parts = line.strip().split("  ")
            if len(parts) == 2:
                word = parts[0].split("(")[0].lower()
                # Remove stress markers (digits) and convert to lowercase
                phones = tuple([re.sub(r'\d+', '', p).lower() for p in parts[1].split()])
                if phones not in rev_dict:
                    rev_dict[phones] = []
                rev_dict[phones].append(word)
    return rev_dict

def decode_phones(phones, rev_dict):
    phones = tuple([p.lower() for p in phones])
    if phones in rev_dict:
        words = rev_dict[phones]
        words.sort(key=len)
        return words[0]
    return "<UNK>"

def process_nus48e():
    print("Processing NUS-48E annotations...")
    dict_path = download_cmudict()
    rev_dict = build_reverse_dict(dict_path)
    
    root = Path("data/raw/nus48e")
    output_dir = Path("data/nus48e_processed")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    clips = []
    
    for singer in sorted(root.iterdir()):
        if not singer.is_dir() or singer.name == "README.txt": continue
        
        for mode in ["sing"]: # We only care about sung lyrics for ASR
            d = singer / mode
            if not d.exists(): continue
            
            for txt_file in d.glob("*.txt"):
                wav_file = txt_file.with_suffix(".wav")
                if not wav_file.exists(): continue
                
                lines = txt_file.read_text(encoding="utf-8").strip().splitlines()
                
                # We group phonemes into ~10 second chunks based on silences
                current_chunk_words = []
                current_chunk_start = 0.0
                current_word_phones = []
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 3: continue
                    start = float(parts[0])
                    end = float(parts[1])
                    ph = parts[2]
                    
                    if current_chunk_start == 0.0:
                        current_chunk_start = start
                        
                    if ph in ("sil", "sp"):
                        if current_word_phones:
                            word = decode_phones(current_word_phones, rev_dict)
                            if word != "<UNK>":
                                current_chunk_words.append(word)
                            current_word_phones = []
                            
                        # If the silence is long enough, end the chunk
                        if end - start > 0.5 and len(current_chunk_words) > 3:
                            text = " ".join(current_chunk_words)
                            # Basic cleanup
                            text = text.replace(" <UNK>", "").replace("<UNK> ", "").replace("<UNK>", "")
                            if text.strip():
                                clips.append({
                                    "audio_path": str(wav_file),
                                    "text": text,
                                    "start": current_chunk_start,
                                    "end": end,
                                    "metadata": {"singer": singer.name, "song": txt_file.stem}
                                })
                            current_chunk_words = []
                            current_chunk_start = end
                    else:
                        current_word_phones.append(ph)
                        
                # Flush remaining
                if current_word_phones:
                    word = decode_phones(current_word_phones, rev_dict)
                    if word != "<UNK>":
                        current_chunk_words.append(word)
                
                if current_chunk_words:
                    text = " ".join(current_chunk_words).strip()
                    if text:
                        clips.append({
                            "audio_path": str(wav_file),
                            "text": text,
                            "start": current_chunk_start,
                            "end": end, # From last line
                            "metadata": {"singer": singer.name, "song": txt_file.stem}
                        })
                        
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(clips, f, indent=2)
    
    print(f"Created {len(clips)} segments from NUS-48E.")
    return manifest_path

if __name__ == "__main__":
    process_nus48e()
