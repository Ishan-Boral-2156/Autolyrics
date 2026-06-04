#!/usr/bin/env python
import math
import struct
import wave
from pathlib import Path


def create_mock_nus48e(root_dir: str):
    root = Path(root_dir)
    singers = {"ADIZ": "train", "ZHIY": "val"}  # One for train, one for val based on config

    sample_rate = 16000
    duration_s = 2.0
    num_samples = int(sample_rate * duration_s)

    for singer in singers:
        singer_dir = root / singer
        sing_dir = singer_dir / "sing"
        lyrics_dir = singer_dir / "lyrics"
        sing_dir.mkdir(parents=True, exist_ok=True)
        lyrics_dir.mkdir(parents=True, exist_ok=True)

        for i in range(2):  # 2 clips per singer
            wav_path = sing_dir / f"{singer}_song_{i}.wav"
            txt_path = lyrics_dir / f"{singer}_song_{i}.txt"

            # Generate sine wave
            with wave.open(str(wav_path), "w") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                for j in range(num_samples):
                    value = int(32767.0 * math.sin(2.0 * math.pi * 440.0 * j / sample_rate))
                    data = struct.pack("<h", value)
                    wav_file.writeframesraw(data)

            # Generate text
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("this is a mock lyric for testing.")

    print(f"Mock dataset created at {root}")


if __name__ == "__main__":
    create_mock_nus48e("data/raw/nus48e")
