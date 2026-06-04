"""AutoLyrics — Dataset loaders for DALI, NUS-48E, Jamendo, and HuggingFace Hub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.logging import get_logger

from data_pipeline.dataset import SingingClip

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
#  Base loader
# ---------------------------------------------------------------------------


class BaseLoader:
    """Base class for all dataset loaders."""

    name: str = "base"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.root = Path(cfg.get("root", f"./data/raw/{self.name}"))
        self.cache_dir = Path(cfg.get("cache_dir", f"./.cache/autolyrics/{self.name}"))

    def load(self) -> list[SingingClip]:
        raise NotImplementedError

    def load_splits(self) -> dict[str, list[SingingClip]]:
        clips = self.load()
        return self._split(clips)

    def _split(self, clips: list[SingingClip]) -> dict[str, list[SingingClip]]:
        import random

        splits_cfg = self.cfg.get("splits", {})
        train_r = splits_cfg.get("train_ratio", 0.8)
        val_r = splits_cfg.get("val_ratio", 0.1)
        random.shuffle(clips)
        n = len(clips)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        train = clips[:n_train]
        val = clips[n_train : n_train + n_val]
        test = clips[n_train + n_val :]
        for c in train:
            c.split = "train"
        for c in val:
            c.split = "val"
        for c in test:
            c.split = "test"
        return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
#  NUS-48E Loader
# ---------------------------------------------------------------------------


class NUS48ELoader(BaseLoader):
    """Loader for the NUS Sung and Spoken Lyrics Corpus (NUS-48E)."""

    name = "nus48e"

    def load(self) -> list[SingingClip]:
        clips: list[SingingClip] = []
        modes = self.cfg.get("modes", ["sing"])
        root = self.root

        if not root.exists():
            logger.warning("NUS-48E root not found at %s", root)
            return clips

        for singer_dir in sorted(root.iterdir()):
            if not singer_dir.is_dir():
                continue
            singer = singer_dir.name
            for mode in modes:
                mode_dir = singer_dir / mode
                if not mode_dir.exists():
                    continue
                for wav_file in sorted(mode_dir.glob("*.wav")):
                    lyric_file = wav_file.with_suffix(".txt")
                    if not lyric_file.exists():
                        lyric_file = singer_dir / "lyrics" / (wav_file.stem + ".txt")
                    text = ""
                    if lyric_file.exists():
                        text = lyric_file.read_text(encoding="utf-8").strip()
                    clips.append(
                        SingingClip(
                            audio_path=str(wav_file),
                            text=text,
                            dataset_name=self.name,
                            metadata={"singer": singer, "mode": mode},
                        )
                    )
        logger.info("NUS-48E: loaded %d clips", len(clips))
        return clips

    def load_splits(self) -> dict[str, list[SingingClip]]:
        clips = self.load()
        splits_cfg = self.cfg.get("splits", {})
        train_singers = set(splits_cfg.get("train_singers", []))
        val_singers = set(splits_cfg.get("val_singers", []))
        test_singers = set(splits_cfg.get("test_singers", []))
        result: dict[str, list[SingingClip]] = {"train": [], "val": [], "test": []}
        for clip in clips:
            singer = clip.metadata.get("singer", "")
            if singer in test_singers:
                clip.split = "test"
                result["test"].append(clip)
            elif singer in val_singers:
                clip.split = "val"
                result["val"].append(clip)
            elif singer in train_singers or not train_singers:
                clip.split = "train"
                result["train"].append(clip)
        logger.info(
            "NUS-48E splits: train=%d, val=%d, test=%d",
            len(result["train"]),
            len(result["val"]),
            len(result["test"]),
        )
        return result


# ---------------------------------------------------------------------------
#  DALI Loader
# ---------------------------------------------------------------------------


class DALILoader(BaseLoader):
    """Loader for the DALI dataset (aligned lyrics for polyphonic songs)."""

    name = "dali"

    def load(self) -> list[SingingClip]:
        clips: list[SingingClip] = []
        root = self.root
        if not root.exists():
            logger.warning("DALI root not found at %s", root)
            return clips

        annotations_dir = root / "annotations"
        audio_dir = root / "audio"
        if not annotations_dir.exists():
            annotations_dir = root
        if not audio_dir.exists():
            audio_dir = root

        min_score = self.cfg.get("filters", {}).get("min_alignment_score", 0.0)
        seg_cfg = self.cfg.get("segmentation", {})
        max_dur = seg_cfg.get("max_duration_s", 25.0)
        min_dur = seg_cfg.get("min_duration_s", 1.5)

        for ann_file in sorted(annotations_dir.glob("*.json")):
            try:
                ann = json.loads(ann_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            song_id = ann_file.stem
            audio_path = None
            for ext in [".mp3", ".wav", ".flac", ".ogg"]:
                candidate = audio_dir / (song_id + ext)
                if candidate.exists():
                    audio_path = str(candidate)
                    break
            if audio_path is None:
                continue

            annotations = ann.get("annotations", ann.get("lines", []))
            for seg in annotations:
                text = seg.get("text", seg.get("words", "")).strip()
                start = float(seg.get("start", seg.get("time", 0)))
                end = float(seg.get("end", start + 5.0))
                score = float(seg.get("score", seg.get("confidence", 1.0)))
                dur = end - start
                if not text or dur < min_dur or dur > max_dur or score < min_score:
                    continue
                clips.append(
                    SingingClip(
                        audio_path=audio_path,
                        text=text,
                        start=start,
                        end=end,
                        dataset_name=self.name,
                        metadata={"song_id": song_id, "score": score},
                    )
                )
        logger.info("DALI: loaded %d clips", len(clips))
        return clips


# ---------------------------------------------------------------------------
#  Jamendo Loader
# ---------------------------------------------------------------------------


class JamendoLoader(BaseLoader):
    """Loader for the Jamendo Lyrics dataset."""

    name = "jamendo"

    def load(self) -> list[SingingClip]:
        clips: list[SingingClip] = []
        root = self.root
        if not root.exists():
            logger.warning("Jamendo root not found at %s", root)
            return clips

        seg_cfg = self.cfg.get("segmentation", {})
        max_dur = seg_cfg.get("max_duration_s", 25.0)
        min_dur = seg_cfg.get("min_duration_s", 1.0)

        for ann_file in sorted(root.glob("**/*.txt")):
            audio_path = None
            for ext in [".mp3", ".wav", ".flac", ".ogg"]:
                candidate = ann_file.with_suffix(ext)
                if candidate.exists():
                    audio_path = str(candidate)
                    break
            if audio_path is None:
                audio_dir = root / "audio"
                for ext in [".mp3", ".wav", ".flac"]:
                    candidate = audio_dir / (ann_file.stem + ext)
                    if candidate.exists():
                        audio_path = str(candidate)
                        break
            if audio_path is None:
                continue

            lines = ann_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            for line in lines:
                parts = line.strip().split(maxsplit=2)
                if len(parts) >= 3:
                    try:
                        start, end = float(parts[0]), float(parts[1])
                        text = parts[2].strip()
                    except ValueError:
                        text = line.strip()
                        start, end = 0.0, 0.0
                else:
                    text = line.strip()
                    start, end = 0.0, 0.0
                dur = end - start if end > start else 0
                if not text:
                    continue
                if dur > 0 and (dur < min_dur or dur > max_dur):
                    continue
                clips.append(
                    SingingClip(
                        audio_path=audio_path,
                        text=text,
                        start=start,
                        end=end,
                        dataset_name=self.name,
                    )
                )
        logger.info("Jamendo: loaded %d clips", len(clips))
        return clips


# ---------------------------------------------------------------------------
#  HuggingFace Hub Loader
# ---------------------------------------------------------------------------


class HFLoader(BaseLoader):
    """Loader for any singing/lyrics dataset on the HuggingFace Hub."""

    name = "hf"

    def load(self) -> list[SingingClip]:
        from datasets import load_dataset

        hf_cfg = self.cfg.get("hf", {})
        dataset_id = hf_cfg.get("dataset_id", "jdmoon/SingingDataset")
        config_name = hf_cfg.get("config_name")
        audio_col = hf_cfg.get("audio_column", "audio")
        text_col = hf_cfg.get("text_column", "text")
        streaming = hf_cfg.get("streaming", False)
        cache_dir = str(self.cache_dir)

        logger.info("Loading HF dataset: %s", dataset_id)
        ds = load_dataset(
            dataset_id,
            name=config_name,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True,
        )

        clips: list[SingingClip] = []
        for split_name in ["train", "validation", "test"]:
            if split_name not in ds:
                continue
            split_ds = ds[split_name]
            for item in split_ds:
                audio = item.get(audio_col, {})
                text = str(item.get(text_col, "")).strip()
                if not text:
                    continue
                audio_path = audio.get("path", "") if isinstance(audio, dict) else ""
                clips.append(
                    SingingClip(
                        audio_path=audio_path,
                        text=text,
                        dataset_name=self.name,
                        split=split_name.replace("validation", "val"),
                        metadata={"hf_dataset": dataset_id},
                    )
                )
        logger.info("HF: loaded %d clips from %s", len(clips), dataset_id)
        return clips

    def load_splits(self) -> dict[str, list[SingingClip]]:
        clips = self.load()
        result: dict[str, list[SingingClip]] = {"train": [], "val": [], "test": []}
        for clip in clips:
            s = clip.split if clip.split in result else "train"
            result[s].append(clip)
        if not result["val"] and not result["test"]:
            return self._split(clips)
        return result


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------

_LOADERS: dict[str, type[BaseLoader]] = {
    "nus48e": NUS48ELoader,
    "dali": DALILoader,
    "jamendo": JamendoLoader,
    "hf": HFLoader,
}


def get_loader(name: str, cfg: dict[str, Any]) -> BaseLoader:
    """Return a dataset loader by name."""
    cls = _LOADERS.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown dataset loader: {name!r}. Available: {list(_LOADERS)}")
    return cls(cfg)
