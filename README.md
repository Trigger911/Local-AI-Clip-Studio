# 🎬 Universal Local AI Video Clip Studio

A high-performance, production-grade desktop application built with Python and `CustomTkinter`, engineered to automate video clipping, subtitle transcription, and highlight detection workflows. Designed from the ground up for local execution on NVIDIA RTX hardware, this tool offloads heavy tensor workloads directly to your GPU to ensure absolute data privacy and zero cloud costs.

---

## ⚡ Core Architecture & Pipeline

The application streamlines video editing by chaining together GPU-accelerated local transcription, contextual AI analysis, and timeline compilation into a single, non-blocking asynchronous workflow.

### **Pipeline Execution Steps:**
1. **Intelligent Ingestion & Auto-Detection:** 
   * Select any high-resolution video file (including `.mp4`, `.mov`, `.mkv`, and Insta360 `.insv` recordings) via the GUI or let the system auto-detect recommended output formats (`.fcpxml` for DaVinci Resolve, `.edl`, `.srt`, `.txt`) and framing presets.
2. **Step 1 — RTX-Accelerated Transcription (`faster-whisper`):** 
   * Offloads audio processing to NVIDIA CUDA cores using OpenAI's `large-v3` model in `float16` precision, generating highly accurate `.srt` transcripts with millisecond-level timestamps.
3. **Step 2 — Contextual Clip Mining & Custom AI Directives (Local LLM):** 
   * Feeds the generated transcript into a local large language model (`qwen2.5:32b`) via Ollama. It evaluates pacing, action sequences, and user-defined prompt templates against your selected **Clip Mining Focus Categories** to isolate exact start times, durations, and reasoning.
4. **Step 3 — Timeline Export & Resume Caching:** 
   * Compiles the results directly into professional editing timeline files. 
   * **Resume Mode:** Automatically checks disk checkpoints (`.srt` and output files) so interrupted sessions or prompt iterations can skip completed steps instantly.

---

## 🏷️ AI Clip Mining Focus Categories & Custom Directives

To help the AI mine your footage for the exact moments you care about, the interface includes:
* **Categorized Focus Tags:** Grouped checkboxes (*Gaming & VR, Action & Extreme, Outdoor & Water, Lifestyle & Pets*) that dynamically inject context into your prompts.
* **Custom AI Prompt Editor:** A built-in template editor allowing you to fine-tune or rewrite the exact instructions sent to the LLM on the fly, tailoring highlight criteria (such as comedy hooks, action climaxes, or dialogue highlights) to your specific project type.

---

## 🧠 Why Ollama & Local Execution?

The application relies on **Ollama** as its local LLM runtime backend. Rather than sending your personal video transcripts to external cloud services or proprietary APIs, everything runs **100% locally on your machine**.

* **Total Data Privacy:** Your video transcripts, audio data, and personal recordings never leave your local machine.
* **Zero Cloud Costs:** Execute complex reasoning and highlight extraction locally without per-token API fees.
* **Model Flexibility & Power:** Designed to interface seamlessly with high-capability local models like `qwen2.5:32b` running on CUDA hardware acceleration.
* **Offline Capability:** Process and clip your footage anywhere, even without an active internet connection.

> **Prerequisite:** Make sure Ollama is installed and running locally on your system (`ollama serve` or via the Windows system tray) so the pipeline can communicate with `http://localhost:11434` during the analysis phase.

---

## 🚀 Getting Started & Requirements

### **1. Prerequisites**
* Python 3.10+
* NVIDIA RTX GPU with up-to-date drivers and CUDA support.
* [Ollama](https://ollama.com/) installed and running locally with the `qwen2.5:32b` model pulled:
  ```bash
  ollama run qwen2.5:32b
