# Local AI Clip Studio (RTX 4090 Optimized)

A 100% local, automated Python GUI application that processes raw Insta360 video files (`.insv`), transcribes audio using **Faster-Whisper** (CUDA-accelerated), queries a local LLM via **Ollama** for highlight selection, and exports a clean timeline directly to **DaVinci Resolve** via `.fcpxml`.

## Features
* **Direct .insv Ingestion**: No manual renaming or separate stitching software required for quick extraction.
* **GPU Accelerated**: Built for high-end NVIDIA hardware utilizing local CUDA runtime DLL linking.
* **Zero Cloud Dependence**: Everything runs entirely offline using Ollama and Faster-Whisper.
* **DaVinci Resolve Integration**: Automatically generates multi-clip timelines structured for vertical (TikTok/Shorts) or landscape edits.

## Prerequisites
1. **Python 3.10+** installed on Windows 11.
2. **Ollama** running locally with your preferred model pulled (e.g., `qwen2.5:32b`).
3. **FFmpeg** installed and added to your system environment variables (required for stream probing).

## Quick Start
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Local-AI-Clip-Studio.git](https://github.com/YourUsername/Local-AI-Clip-Studio.git)
   cd Local-AI-Clip-Studio
