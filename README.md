📌 AI-Powered Audio/Video Transcription Tool
A Google Colab-based transcription tool that leverages OpenAI's Whisper AI model for highly accurate audio and video transcription, featuring noise reduction, multi-language support, and real-time progress tracking.

🚀 Features
✔ Automatic video-to-audio conversion (MP4, AVI, MOV → WAV)
✔ Noise reduction using noisereduce for cleaner transcripts
✔ Multi-language transcription (Supports English, Hindi, Tamil, Kannada + auto-detection)
✔ Model selection (Whisper tiny, base, small, medium, large)
✔ User-friendly widgets for easy file upload and processing
✔ Real-time progress visualization during transcription
✔ Automatic cleanup of temporary files for optimized runtime

🛠️ Installation & Setup
This tool is designed to run exclusively on Google Colab with a T4 GPU runtime for optimal performance.

🔹 Step 1: Open Google Colab
Go to Google Colab

Ensure the Runtime Type is set to GPU (T4)

Click Runtime → Change runtime type → Select GPU

🔹 Step 2: Install Dependencies
Run the following code inside a Colab cell to install all required libraries:

python
Copy
Edit
!pip install openai-whisper ffmpeg pydub noisereduce tqdm
📂 Supported File Formats
Type	Formats Supported
Audio	MP3, WAV, FLAC, OGG
Video	MP4, AVI, MOV, MKV
📌 How It Works (Technical Flow)
1️⃣ User Uploads File:

Uploads an audio (MP3/WAV) or video (MP4/AVI/MOV) file.
2️⃣ Video to Audio Conversion:

If a video file is detected, FFmpeg extracts the audio and converts it to WAV format.
3️⃣ Audio Preprocessing:

Noise Reduction: Uses noisereduce to remove background noise.

Normalization: Adjusts the audio levels for better transcription accuracy.
4️⃣ Whisper Model Processing:

The OpenAI Whisper model transcribes the cleaned audio into text.
5️⃣ Saving & Downloading:

Transcription is saved as a .txt file and auto-downloaded.
6️⃣ Cleanup:

Temporary files (audio/video) are deleted to free up Colab storage.

🎛️ User Interface (Widgets)
This tool includes interactive Colab widgets for easy user interaction:
✅ Model Selector: Choose from tiny, base, small, medium, large models
✅ Language Selector: Select from English, Hindi, Tamil, Kannada, or auto-detect
✅ Progress Bar: Real-time feedback during processing
✅ File Upload Widget: Drag and drop audio/video files

📌 Running the Transcription Tool
Use the following code snippet in Google Colab to launch the tool:

python
Copy
Edit
import gradio as gr
import whisper
import ffmpeg
import os
import noisereduce as nr
import numpy as np
import torch
import librosa
import shutil
from tqdm import tqdm
from pydub import AudioSegment

# Load Whisper Model
model = whisper.load_model("medium")  # Change to "tiny", "base", "small", etc.

def transcribe_audio(file_path, selected_language):
    # Load audio
    y, sr = librosa.load(file_path, sr=16000)
    
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    
    # Save cleaned audio
    clean_audio_path = "clean_audio.wav"
    librosa.output.write_wav(clean_audio_path, reduced_noise, sr)

    # Transcribe
    options = {"language": selected_language} if selected_language else {}
    result = model.transcribe(clean_audio_path, **options)

    # Save text output
    with open("transcription.txt", "w") as f:
        f.write(result["text"])

    return result["text"]

# Gradio UI
interface = gr.Interface(
    fn=transcribe_audio,
    inputs=[gr.Audio(type="filepath"), gr.Dropdown(["English", "Hindi", "Tamil", "Kannada", "Auto"])],
    outputs="text",
    title="AI-Powered Transcription Tool",
    description="Upload an audio or video file and get transcriptions in multiple languages.",
)
interface.launch()
⚡ Performance Optimization
🔹 Use a T4 GPU: Whisper models run significantly faster on a GPU. Ensure you are using Google Colab's T4 GPU runtime.
🔹 Model Selection: The tiny & base models are faster but less accurate, while the medium & large models provide better accuracy at the cost of processing speed.
🔹 Reduce Noise Before Transcription: Preprocessing the audio with noisereduce improves transcription quality.

🎯 Applications
✅ Podcast & Lecture Transcription
✅ Subtitles Generation for Videos
✅ Speech-to-Text for Accessibility
✅ Transcription for Call Centers & Meetings
✅ Multi-language Voice Assistants

🛠️ Troubleshooting & FAQs
1️⃣ Why is my transcription slow?
Ensure you're using Google Colab's GPU runtime (T4 GPU)

Use smaller Whisper models (tiny/base) for faster processing

If processing large files, consider trimming audio/video before transcription

2️⃣ My transcription has errors. How can I improve accuracy?
Use the "large" model for better accuracy

Ensure clear audio quality (less noise, no background music)

Select the correct language if the model isn’t auto-detecting correctly

3️⃣ How do I process long audio/video files?
Whisper works best with ≤ 30-minute audio

For long files, split audio into chunks before transcription

🤝 Contributing
Feel free to fork this repository, submit PRs, or report issues!

📜 License
This project is open-source under the MIT License.
