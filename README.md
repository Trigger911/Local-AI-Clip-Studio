# 🚀 Pipeline Architecture & Local AI Requirements

**Local AI Clip Studio** is a production-grade, RTX-accelerated desktop application designed to fully automate video clipping and highlight detection workflows for high-resolution content (including Insta360 `.insv` footage and vertical mobile recordings). 

---

## ⚡ 1. The Automated Pipeline (`Auto-Cut Pipeline`)

The application streamlines video editing by chaining together GPU-accelerated local transcription, contextual AI analysis, and timeline compilation into a single, non-blocking asynchronous workflow.

### **Pipeline Execution Steps:**
1. **Intelligent Ingestion & Auto-Detection:** 
   * Select any video file via the GUI or let the system auto-detect output formats (`.fcpxml` for DaVinci Resolve, `.edl`, `.srt`, `.txt`) and framing presets (**Vertical 9:16 Shorts/TikTok** vs. **Full 360° Source**).
2. **Step 1 — RTX-Accelerated Transcription (`faster-whisper`):** 
   * Offloads audio processing to NVIDIA CUDA cores using OpenAI's `large-v3` model in `float16` precision, generating highly accurate `.srt` transcripts with millisecond-level timestamps.
3. **Step 2 — Contextual Highlight Analysis (Local LLM):** 
   * Feeds the generated transcript into a local large language model to evaluate narrative pacing, action sequences, or key commentary, isolating the exact start times and durations for highlight clips.
4. **Step 3 — Timeline Export & Resume Caching:** 
   * Compiles the results directly into professional editing timeline files. 
   * **Resume Mode:** Automatically checks disk checkpoints (`.srt` and output files) so interrupted sessions or prompt iterations can skip completed steps instantly.

---

## 🧠 2. Why Ollama is Required

The application relies on **Ollama** as its local LLM runtime backend. Rather than sending your video transcripts to external cloud services or proprietary APIs, everything runs **100% locally on your machine**.

### **Key Reasons for Ollama Integration:**
* **Total Data Privacy:** Your video transcripts, audio data, and personal recordings never leave your local machine.
* **Zero Cloud Costs:** Execute complex reasoning and highlight extraction locally without per-token API fees.
* **Model Flexibility & Power:** Designed to interface seamlessly with high-capability local models like `qwen2.5:32b` running on CUDA hardware acceleration for rapid response times.
* **Offline Capability:** Process and clip your footage anywhere, even without an active internet connection.

> **Prerequisite:** Make sure Ollama is installed and running locally on your system (`ollama serve` or via the Windows system tray) so the pipeline can communicate with `http://localhost:11434` during the analysis phase.
