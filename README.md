🎥 Multimodal Video Captioning and Analysis Pipeline
This repository contains a powerful end-to-end Python pipeline for multimodal video analysis using state-of-the-art deep learning models. It performs:

Scene Description (image captioning via GIT)

Object Detection (YOLOv5n)

Speech Transcription (Whisper-tiny)

Video Augmentation with overlaid captions, object boxes, and audio

⚡ The pipeline downloads a YouTube video, processes it frame-by-frame, and generates a final annotated video with both visual and speech information using PyTorch, ONNX, and ffmpeg.

🚀 Features
🔊 Audio Transcription using OpenAI Whisper

🖼️ Scene Description using Microsoft GIT

📦 Object Detection using Ultralytics YOLOv5 (PyTorch + ONNX + INT8 Quantization)

🧠 Model Benchmarking for speed and memory footprint

🎞️ Frame-wise Annotation using OpenCV with speech and scene overlays

📦 ONNX Export & Quantization for efficient inference

🧪 Runs fully in Google Colab with GPU

🧰 Requirements
The following packages are installed automatically in the script:

PyTorch + TorchVision + Torchaudio

OpenAI Whisper

Transformers + Tokenizers

ONNX + ONNXRuntime

YOLOv5 (via Ultralytics + TorchHub)

OpenCV + yt-dlp + ffmpeg

📦 Setup Instructions
✅ Recommended: Run in Google Colab with GPU runtime.

1. Install All Dependencies
python

 Run the script or notebook directly
 It will:
 Uninstall conflicts
Install PyTorch, Whisper, YOLO, etc.
Install ffmpeg for audio/video processing
📌 Make sure your runtime supports CUDA (GPU). The script checks and uses CUDA if available.

🧪 Inference Pipeline
The pipeline processes a video in two main parts:

🔧 Part 1: Model Installation & Benchmarking
Installs and verifies all dependencies

Loads:

GIT (git-base) for scene description

YOLOv5n via Ultralytics and TorchHub for object detection

Whisper-tiny for speech-to-text

Benchmarks latency and memory size for each model

Exports YOLOv5 to ONNX + performs dynamic INT8 quantization

🎬 Part 2: Video Analysis & Annotation
Downloads a YouTube video

Extracts speech for transcription

Processes each video frame:

Scene description caption (top left)

Object detection with bounding boxes (top left)

Overlaid spoken text (bottom)

Merges output video and cleaned audio using ffmpeg

🖼️ Output
Final output is a video:

🎯 Annotated with object boxes, scene captions, and spoken dialogue

🎧 Merged with cleaned audio

✨ Ready for visualization, research, or deployment

## 📁 Directory Structure

<pre>
├── optimized_models/
│   ├── yolov5n_fp32.onnx          # YOLOv5n exported in ONNX FP32
│   └── yolov5n_int8_dynamic.onnx  # YOLOv5n quantized to INT8 ONNX
├── inf_vid.mp4                    # Downloaded YouTube video for analysis
├── cap_vid_final_audio.mp4        # Final output with scene, object, and speech overlays
└── dummy_audio_p1.wav             # Dummy audio used for benchmarking Whisper
</pre>

---

## 📊 Benchmarking Example

| Model                 | Avg Latency (ms) | Size (MB) |
|-----------------------|------------------|-----------|
| GIT (PT)              | ~XX              | ~XXX      |
| YOLOv5n (PT)          | ~XX              | ~XX       |
| YOLOv5n (ONNX FP32)   | ~XX              | ~XX       |
| YOLOv5n (ONNX INT8)   | ~XX              | ~XX       |
| Whisper-tiny (PT)     | ~XX              | ~XX       |

> 📌 *Values depend on runtime hardware (e.g., Colab GPU, local CPU) and batch size.*

