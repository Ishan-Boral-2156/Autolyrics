import os

import gradio as gr
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

# Force inject FFmpeg into PATH dynamically to bypass Windows terminal restart issues
ffmpeg_bin = r"C:\Users\ishan\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
if os.path.exists(ffmpeg_bin) and ffmpeg_bin not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]


MODEL_NAME = "openai/whisper-tiny"
LORA_PATH = "runs/autolyrics_complete/lora_decoder/best"

print("Loading Whisper model...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading Base Whisper model...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
dtype = torch.float16 if device == "cuda:0" else torch.float32

base_model_pure = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(
    device
)

print("Loading Fine-Tuned LoRA model...")
# Load a second instance for the adapter so we don't pollute the base model
base_model_for_lora = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=dtype
).to(device)
model_finetuned = PeftModel.from_pretrained(base_model_for_lora, LORA_PATH)
model_finetuned = model_finetuned.merge_and_unload()

# Create HuggingFace Pipelines for both
pipe_base = pipeline(
    "automatic-speech-recognition",
    model=base_model_pure,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    device=0 if device == "cuda:0" else -1,
)

pipe_finetuned = pipeline(
    "automatic-speech-recognition",
    model=model_finetuned,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    device=0 if device == "cuda:0" else -1,
)


def transcribe_song(audio, model_choice):
    if audio is None:
        return "Please upload an audio file."

    sample_rate, audio_array = audio

    # Convert to float32
    import numpy as np

    audio_array = audio_array.astype(np.float32)

    # If stereo (2 channels), convert to mono
    if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
        audio_array = audio_array.mean(axis=1)

    # Normalize from int16 to float32 range [-1, 1]
    if np.abs(audio_array).max() > 1.0:
        audio_array = audio_array / 32768.0

    print(f"Transcribing audio array using {model_choice}...")
    active_pipe = pipe_finetuned if "Fine-Tuned" in model_choice else pipe_base

    # Run inference through the pipeline using the raw array
    result = active_pipe(
        {"sampling_rate": sample_rate, "raw": audio_array},
        return_timestamps=True,
        generate_kwargs={
            "language": "english",
            "task": "transcribe",
            "condition_on_prev_tokens": False,
            "repetition_penalty": 1.2,
        },
    )

    # Format as karaoke-style lyrics if timestamps are returned
    if "chunks" in result:
        formatted_lyrics = []
        for chunk in result["chunks"]:
            start = chunk["timestamp"][0]
            end = chunk["timestamp"][1]
            text = chunk["text"].strip()

            def format_time(sec):
                if sec is None:
                    return "??:??"
                m, s = divmod(int(sec), 60)
                return f"{m:02d}:{s:02d}"

            formatted_lyrics.append(f"[{format_time(start)} - {format_time(end)}] {text}")
        return "\n".join(formatted_lyrics)

    return result["text"].strip()


# Create a sleek, modern Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown(
        """
        # 🎵 AutoLyrics: Singing-Aware ASR
        Upload any song (even full 3-5 minute tracks), and our fine-tuned Whisper-tiny model will transcribe the lyrics!
        *This model has been specifically adapted to understand the acoustic properties of singing vocals.*
        """
    )

    with gr.Row():
        model_choice = gr.Radio(
            choices=["Base Whisper (Pop Optimized)", "Fine-Tuned (A-cappella Optimized)"],
            value="Base Whisper (Pop Optimized)",
            label="Select Model Version",
        )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Upload Song (MP3/WAV)")
            submit_btn = gr.Button("Transcribe Lyrics", variant="primary")

        with gr.Column():
            lyrics_output = gr.Textbox(label="Transcribed Lyrics", lines=10)

    submit_btn.click(fn=transcribe_song, inputs=[audio_input, model_choice], outputs=lyrics_output)

if __name__ == "__main__":
    print("\nStarting AutoLyrics Web UI...")
    demo.launch(share=False)
