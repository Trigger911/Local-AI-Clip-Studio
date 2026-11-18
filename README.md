# 🎬 Universal Local AI Video Clip Studio

A high-performance, production-grade desktop application built with Python and `CustomTkinter`, engineered to automate video clipping, subtitle transcription, and highlight detection workflows. Designed from the ground up for local execution on NVIDIA RTX hardware, this tool offloads heavy tensor workloads directly to your GPU to ensure absolute data privacy and zero cloud costs.

---

## ⚡ Core Architecture & Pipeline

The application streamlines video editing by chaining together GPU-accelerated local transcription, contextual AI analysis, and timeline compilation into a single, non-blocking asynchronous workflow.

### **Pipeline Execution Steps:**
1. **Intelligent Ingestion & Auto-Detection:** 
   * Select any high-resolution video file (including `.mp4`, `.mov`, `.mkv`, and Insta360 `.insv` recordings) via the GUI or let the system auto-detect recommended output formats (`.fcpxml` for DaVinci Resolve, `.edl`, `.srt`, `.txt`) and framing presets.
2. **Step 1 — Hybrid RTX/CPU Transcription (`faster-whisper`):** 
   * Offloads neural audio processing to NVIDIA CUDA cores using OpenAI's `large-v3` model in `float16` precision, while concurrent CPU thread pooling handles fast audio decoding without bottlenecking. Generates precise `.srt` transcripts with millisecond-level timestamps.
3. **Step 2 — Contextual Clip Mining & Local LLM Analysis (Ollama):** 
   * Feeds transcripts into your selected local LLM (`qwen2.5:32b`, `qwen2.5:14b`, `llama3:8b`, or `mistral:7b`) via Ollama. It evaluates pacing and action sequences against your checked **Clip Mining Focus Categories** (dynamically injected into the prompt template) to isolate exact start times, durations, and reasoning.
4. **Step 3 — Timeline Export & Resume Caching:** 
   * Compiles the results directly into professional editing timeline files. 
   * **Resume Mode:** Automatically checks disk checkpoints (`.srt` and output files) so interrupted sessions or prompt iterations can skip completed steps instantly.

---

## 🧠 Local LLM Performance & Hardware Inspector

The application features a built-in **LLM Inspector** that lets you instantly switch models and inspect their hardware footprint, speeds, pros, and cons right in the UI:

* **`qwen2.5:32b`**: Elite heavyweight reasoning model. Unmatched context comprehension and detailed clip justifications, but requires ~20GB+ VRAM and runs slower.
* **`qwen2.5:14b`**: Balanced sweet-spot model. Offers near-32B quality at nearly double the speed with a moderate ~10-12GB VRAM footprint.
* **`llama3:8b`**: Blazing fast lightweight workhorse. Perfect for rapid iterative highlight testing with minimal ~6GB VRAM usage.
* **`mistral:7b`**: Agile instruction-following model optimized for highly structured timestamp formatting and rapid response times.

---

## 🏷️ AI Clip Mining Focus Categories & Dynamic Prompting

To help the AI mine your footage for the exact moments you care about, the interface includes:
* **Categorized Focus Tags:** Grouped checkboxes (*Gaming & VR, Action & Extreme, Outdoor & Water, Lifestyle & Pets*) with professional editing definitions.
* **Dynamic Prompt Injection:** Checking hobby categories automatically populates the `{criteria_details}` block in your custom prompt template, ensuring the AI targets specific action beats, audio spikes, or tricks.

---

## 🔒 Why Ollama & Local Execution?

The application relies on **Ollama** as its local LLM runtime backend. Rather than sending your personal video transcripts to external cloud services or proprietary APIs, everything runs **100% locally on your machine**.

* **Total Data Privacy:** Your video transcripts, audio data, and personal recordings never leave your local machine.
* **Zero Cloud Costs:** Execute complex reasoning and highlight extraction locally without per-token API fees.
* **Offline Capability:** Process and clip your footage anywhere, even without an active internet connection.

> **Prerequisite:** Make sure Ollama is installed and running locally on your system (`ollama serve` or via the Windows system tray) so the pipeline can communicate with `http://localhost:11434` during the analysis phase.

---

## 🚀 Getting Started & Requirements

### **1. Prerequisites**
* Python 3.10+
* NVIDIA RTX GPU with up-to-date drivers and CUDA support.
* [Ollama](https://ollama.com/) installed and running locally with your desired model pulled (e.g., `ollama run qwen2.5:32b`).

### **2. Installation**
Clone the repository and install the required Python dependencies:

```bash
git clone [https://github.com/your-username/local-ai-clip-studio.git](https://github.com/your-username/local-ai-clip-studio.git)
cd local-ai-clip-studio
pip install customtkinter faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
