# 🎵 AutoLyrics: Singing-Aware ASR Pipeline

> A robust, end-to-end Automatic Speech Recognition (ASR) pipeline optimizing OpenAI's Whisper model for singing-voice transcription via Parameter-Efficient Fine-Tuning (PEFT) and LoRA.

AutoLyrics addresses the unique challenges of singing-voice domain transcription—such as prolonged phonemes, dramatic pitch variation, and heavy background instrumentation. This project demonstrates an end-to-end Machine Learning Engineering lifecycle: from raw audio ingestion and spectrogram processing, to LoRA fine-tuning, and finally deploying a dynamic React/Gradio Web UI for real-time A/B testing.

### Performance on NUS-48E Sung Corpus

Evaluated on a rigorous, **singer-disjoint test split** (4 unseen singers) using the real NUS-48E corpus. By scaling the pipeline to `whisper-small` (244M parameters), the LoRA adaptation achieved a massive **38.9% relative reduction** in Word Error Rate.

| Model Variant | Word Error Rate (WER) | Character Error Rate (CER) |
|---------------|-----------------------|----------------------------|
| Whisper-small (Zero-shot) | 12.72% | 6.04% |
| Whisper-small + PEFT LoRA | **7.76%** | **4.32%** |

*(Note: The codebase defaults to `whisper-tiny` for rapid local testing and fast inference on consumer GPUs, but can be scaled to `whisper-small` simply by modifying `MODEL_NAME` in the config.)*

---

## 🎯 Key Technical Highlights

1. **End-to-End ML Pipeline:** Built a fully modular pipeline handling data preprocessing (`torchaudio`), model training (`PyTorch`, `HuggingFace`), evaluation (WER/CER metrics), and deployment.
2. **Parameter-Efficient Fine-Tuning (PEFT):** Implemented Low-Rank Adaptation (LoRA) to fine-tune OpenAI's Whisper models on a single consumer GPU, optimizing acoustic features for prolonged singing vowels.
3. **Interactive A/B Testing Web UI:** Engineered a modern Gradio web application that dynamically loads both the Base Whisper model (Generalist) and the LoRA Fine-Tuned model (A-cappella Specialist) into VRAM via FP16 precision. Users can toggle between models on-the-fly to visually analyze transcription accuracy across different musical genres.
4. **Karaoke-Style Time Synchronization:** Leveraged Whisper's timestamp prediction capabilities to chunk and align text inference, outputting granular, real-time karaoke syncs (`[00:15 - 00:20] I got my mind set on you`).
5. **Domain Mismatch Analysis:** Successfully evaluated the effects of Catastrophic Forgetting in LLMs/ASRs. The LoRA model demonstrated superior transcription on acoustic/a-cappella datasets (NUS-48E) but exhibited predictable hallucination loops on polyphonic pop tracks, establishing a clear use-case for domain-matched training data.

---

## 🚀 Getting Started

The project is streamlined into a command-line interface for executing the complete machine learning pipeline and running real-time demos.

### Prerequisites

Ensure you have Python 3.10+ and FFmpeg installed.

```bash
# Clone the repository
git clone https://github.com/Ishan-Boral-2156/Autolyrics.git
cd Autolyrics

# Create and activate a virtual environment
python -m venv .venv
# On Windows use: .venv\Scripts\activate
# On Mac/Linux use: source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🛠️ Usage

### 1. Interactive Web Application (Gradio)

For a user-friendly experience, we provide a full web interface that allows you to upload any song, toggle between the Base and Fine-Tuned models for A/B testing, and receive **Karaoke-style synchronized lyrics**.

```bash
python app.py
```

This will launch a local web server (typically at `http://127.0.0.1:7860`). Upload a song file (MP3/WAV) to see the model output timestamped lyrics in real-time.

### 2. End-to-End Model Training Pipeline

The entire machine learning pipeline—from data ingestion to model training and automated PDF report generation—is orchestrated through a single entry point.

```bash
python run_complete.py
```

Running this script will execute the workflow, providing detailed logs and generating an evaluation report detailing WER/CER metrics and benchmarking results.

### 3. Real-Time CLI Inference

You can also run time-synchronized inference directly from the command line:

```bash
python demo_realtime.py "path/to/song.wav"
```

---

## 📂 Repository Structure

- `core/` - Shared utilities, configurations, and central logic.
- `data_pipeline/` - Dataset ingestion, `torchaudio` preprocessing, and augmentation.
- `training/` - Model definitions, Whisper integration, and LoRA logic.
- `evaluation/` - Benchmarking, WER/CER calculation, and automated reporting.
- `app.py` - Gradio Web UI with dual-model VRAM loading and Karaoke chunking.

---

## 🛠️ Built With
* **PyTorch & Torchaudio:** Core deep learning math, tensors, and real-time audio sample-rate conversion.
* **HuggingFace (Transformers & PEFT):** Whisper model architecture and LoRA adapters.
* **Gradio:** React-based web interface generation.
* **FFmpeg:** Raw audio decoding.

## License

MIT — see [LICENSE](LICENSE).