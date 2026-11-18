import sys
import subprocess
import os
import site

# --- SILENCE HUGGING FACE WARNINGS ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# --- AUTOMATIC NVIDIA CUDA DLL PATH LINKER (GLOBAL + USER) ---
search_paths = site.getsitepackages() + [site.getusersitepackages()]
for site_dir in search_paths:
    cublas_bin = os.path.join(site_dir, "nvidia", "cublas", "bin")
    cudnn_bin = os.path.join(site_dir, "nvidia", "cudnn", "bin")
    if os.path.exists(cublas_bin):
        try:
            os.add_dll_directory(cublas_bin)
            print(f"[VERBOSE] Added CUDA DLL path: {cublas_bin}")
        except Exception:
            pass
    if os.path.exists(cudnn_bin):
        try:
            os.add_dll_directory(cudnn_bin)
            print(f"[VERBOSE] Added CUDA DLL path: {cudnn_bin}")
        except Exception:
            pass

# --- AUTOMATIC DEPENDENCY CHECKER ---
REQUIRED_PACKAGES = {
    "customtkinter": "customtkinter",
    "faster-whisper": "faster-whisper",
    "requests": "requests",
    "av": "av<19",
    "nvidia_cublas_cu12": "nvidia-cublas-cu12",
    "nvidia_cudnn_cu12": "nvidia-cudnn-cu12"
}

def check_and_install_dependencies():
    print("[VERBOSE] Starting automatic dependency check...")
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module_name.replace("-", "_"))
            print(f"[VERBOSE] Dependency check OK: '{module_name}' is already installed.")
        except ImportError:
            print(f"[VERBOSE] Missing or incompatible package '{pip_name}'. Installing automatically via pip...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                print(f"[VERBOSE] Successfully installed {pip_name}!")
            except Exception as e:
                print(f"[ERROR] Failed to auto-install {pip_name}: {e}")
                sys.exit(1)

check_and_install_dependencies()

# --- STANDARD IMPORTS ---
import json
import threading
import requests
import customtkinter as ctk
from tkinter import filedialog
from faster_whisper import WhisperModel

# --- UI APPEARANCE ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

OLLAMA_URL = "http://localhost:11434/api/generate"

class AutoCutApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        print("[VERBOSE] Initializing Local AI Clip Studio GUI...")

        self.title("Local AI Clip Studio (RTX 4090 - Direct INSV Support)")
        self.geometry("680x600")
        self.resizable(False, False)

        self.video_path = ctk.StringVar(value="No video selected...")
        self.mode_var = ctk.StringVar(value="TikTok / Vertical (Auto-Crop 9:16)")
        self.model_var = ctk.StringVar(value="qwen2.5:32b")
        self.padding_var = ctk.DoubleVar(value=0.5)

        self._build_ui()
        print("[VERBOSE] GUI initialization complete.")

    def _build_ui(self):
        header = ctk.CTkLabel(self, text="Local AI Video Clipper & Resolve Builder", font=("Segoe UI", 22, "bold"))
        header.pack(pady=(20, 5))
        
        sub = ctk.CTkLabel(self, text="100% Local • Direct .insv & .mp4 Processing", text_color="gray")
        sub.pack(pady=(0, 15))

        # 1. File Selection Frame
        file_frame = ctk.CTkFrame(self)
        file_frame.pack(fill="x", padx=25, pady=8)

        ctk.CTkButton(file_frame, text="Select Video File", command=self.browse_file, width=140).pack(side="left", padx=12, pady=12)
        ctk.CTkLabel(file_frame, textvariable=self.video_path, anchor="w", wraplength=460).pack(side="left", padx=5)

        # 2. Settings Row
        settings_frame = ctk.CTkFrame(self)
        settings_frame.pack(fill="x", padx=25, pady=8)

        ctk.CTkLabel(settings_frame, text="Output Style:", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, padx=15, pady=(10, 2), sticky="w")
        mode_menu = ctk.CTkOptionMenu(
            settings_frame, 
            variable=self.mode_var,
            values=["TikTok / Vertical (Auto-Crop 9:16)", "Full 360° Timeline (Keep Equirectangular)", "Standard Landscape (16:9)"],
            width=280
        )
        mode_menu.grid(row=1, column=0, padx=15, pady=(0, 12), sticky="w")

        ctk.CTkLabel(settings_frame, text="Local AI Model:", font=("Segoe UI", 13, "bold")).grid(row=0, column=1, padx=15, pady=(10, 2), sticky="w")
        model_entry = ctk.CTkEntry(settings_frame, textvariable=self.model_var, width=220)
        model_entry.grid(row=1, column=1, padx=15, pady=(0, 12), sticky="w")

        # 3. AI Prompt Instructions
        prompt_frame = ctk.CTkFrame(self)
        prompt_frame.pack(fill="x", padx=25, pady=8)

        ctk.CTkLabel(prompt_frame, text="What should the AI look for? (Edit if needed):", font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=15, pady=(10, 4))
        self.prompt_box = ctk.CTkTextbox(prompt_frame, height=65, font=("Segoe UI", 12))
        self.prompt_box.pack(fill="x", padx=15, pady=(0, 12))
        self.prompt_box.insert("1.0", "Find the funniest moments, best talking highlights, high-energy reactions, or interesting things happening. Merge adjacent segments from the same thought.")

        # 4. Run Button
        self.run_btn = ctk.CTkButton(
            self, text="Generate Highlights & Resolve Timeline", 
            font=("Segoe UI", 15, "bold"), height=40, fg_color="#1f6aa5", hover_color="#144870",
            command=self.start_processing
        )
        self.run_btn.pack(fill="x", padx=25, pady=10)

        # 5. Live Status Console
        self.log_box = ctk.CTkTextbox(self, height=120, font=("Consolas", 11), state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=25, pady=(0, 15))

    def log(self, msg):
        print(f"[UI-LOG] {msg}")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Insta360 .insv or Video File",
            filetypes=[("Insta360 & Video Files", "*.insv *.mp4 *.mov *.mkv"), ("All Files", "*.*")]
        )
        if path:
            self.video_path.set(path)
            print(f"[VERBOSE] User selected video path: {path}")

    def start_processing(self):
        path = self.video_path.get()
        if not os.path.exists(path):
            self.log("[!] Please select a valid video file first.")
            print("[VERBOSE ERROR] Process aborted: Invalid or missing file path.")
            return
        print(f"[VERBOSE] Starting background processing thread for file: {path}")
        self.run_btn.configure(state="disabled", text="Processing on RTX 4090...")
        threading.Thread(target=self.run_pipeline, args=(path,), daemon=True).start()

    def get_video_info(self, path):
        print(f"[VERBOSE] Running ffprobe to extract video stream metrics for: {path}")
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path]
        data = json.loads(subprocess.check_output(cmd))
        v = next(s for s in data["streams"] if s["codec_type"] == "video")
        if "/" in v["r_frame_rate"]:
            num, den = map(int, v["r_frame_rate"].split("/"))
            fps = round(num / den, 3)
        else:
            fps = float(v["r_frame_rate"])
        duration = float(v.get("duration", 0)) or float(data["format"].get("duration", 0))
        print(f"[VERBOSE] Video Info Extracted -> FPS: {fps}, Duration: {duration}s, Res: {v['width']}x{v['height']}")
        return fps, duration, int(v["width"]), int(v["height"])

    def fmt_srt(self, sec):
        h, r = divmod(sec, 3600)
        m, s = divmod(r, 60)
        ms = int((s - int(s)) * 1000)
        return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"

    def run_pipeline(self, path):
        try:
            base = os.path.splitext(path)[0]
            
            # Step 1: Whisper Transcription
            self.log("1/4 Loading Faster-Whisper (large-v3) onto RTX 4090 (CUDA)...")
            print("[VERBOSE] Initializing WhisperModel('large-v3', device='cuda', compute_type='float16')...")
            model = WhisperModel("large-v3", device="cuda", compute_type="float16")
            
            self.log("1/4 Transcribing audio stream (VAD filter enabled)...")
            print(f"[VERBOSE] Beginning transcription stream on file: {path}")
            segments, info = model.transcribe(path, vad_filter=True)

            lines, srt = [], []
            seg_count = 0
            for i, seg in enumerate(segments, 1):
                seg_count += 1
                s, e, t = round(seg.start, 2), round(seg.end, 2), seg.text.strip()
                lines.append(f"({s}s - {e}s): {t}")
                srt.append(f"{i}\n{self.fmt_srt(s)} --> {self.fmt_srt(e)}\n{t}\n")

            print(f"[VERBOSE] Transcription completed. Total speech segments found: {seg_count}")
            if not lines:
                self.log("[!] No speech detected in video. Try lowering VAD sensitivity.")
                print("[VERBOSE WARNING] Zero speech segments returned by faster-whisper.")
                return

            srt_path = base + ".srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(srt))
            self.log(f"    -> Captions saved: {os.path.basename(srt_path)}")

            # Step 2: Ollama Query
            ai_model = self.model_var.get().strip()
            user_goal = self.prompt_box.get("1.0", "end").strip()
            self.log(f"2/4 Querying local Ollama model ({ai_model})...")
            print(f"[VERBOSE] Sending HTTP POST request to Ollama endpoint: {OLLAMA_URL}")

            prompt = (
                f"{user_goal}\n"
                "Return ONLY a valid JSON array of objects with 'start' and 'end' float keys. "
                "Do not output markdown blocks or extra text:\n\n" + "\n".join(lines)
            )

            resp = requests.post(OLLAMA_URL, json={"model": ai_model, "prompt": prompt, "stream": False})
            print(f"[VERBOSE] Ollama HTTP Response Status Code: {resp.status_code}")
            
            resp_json = resp.json()
            raw_text = resp_json.get("response", "[]").strip()
            print(f"[VERBOSE] Raw Ollama Response Text:\n{raw_text}")

            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:].strip()

            raw = json.loads(raw_text)
            clips = raw if isinstance(raw, list) else next((v for v in raw.values() if isinstance(v, list)), [])
            self.log(f"    -> AI selected {len(clips)} highlight clips.")
            print(f"[VERBOSE] Parsed {len(clips)} valid highlight intervals from Ollama JSON.")

            # Step 3: FCPXML Generation
            self.log("3/4 Building DaVinci Resolve Timeline (.fcpxml)...")
            fps, total_dur, src_w, src_h = self.get_video_info(path)
            
            mode = self.mode_var.get()
            if "TikTok" in mode:
                out_w, out_h = 1080, 1920
            elif "Full 360" in mode:
                out_w, out_h = src_w, src_h
            else:
                out_w, out_h = 1920, 1080

            fdur = f"100/{int(round(fps * 100))}s"
            uri = "file:///" + os.path.abspath(path).replace("\\", "/")

            xml_clips, offset = [], 0.0
            pad = self.padding_var.get()
            for idx, c in enumerate(clips):
                s = max(0.0, float(c["start"]) - pad)
                e = min(total_dur, float(c["end"]) + pad)
                dur = round(e - s, 3)
                if dur <= 0.5:
                    continue
                xml_clips.append(
                    f'    <asset-clip name="Highlight_{idx+1}" ref="r1" offset="{offset:.3f}s" '
                    f'start="{s:.3f}s" duration="{dur:.3f}s" tcFormat="NDF"/>'
                )
                offset += dur

            fcpxml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.9">
  <resources>
    <format id="r0" frameDuration="{fdur}" width="{out_w}" height="{out_h}"/>
    <asset id="r1" name="{os.path.basename(path)}" src="{uri}" start="0s" duration="{total_dur:.3f}s" hasVideo="1"/>
  </resources>
  <library>
    <event name="AI Highlights">
      <project name="AutoCut">
        <sequence format="r0" tcStart="0s" tcFormat="NDF">
          <spine>
{chr(10).join(xml_clips)}
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>'''

            out_xml = base + "_AutoCut.fcpxml"
            with open(out_xml, "w", encoding="utf-8") as f:
                f.write(fcpxml)

            self.log(f"4/4 DONE! Import this into Resolve:\n    {out_xml}")
            print(f"[VERBOSE] Success! FCPXML successfully written to: {out_xml}")
            os.startfile(os.path.dirname(out_xml))

        except Exception as err:
            import traceback
            err_details = traceback.format_exc()
            self.log(f"[ERROR] {str(err)}")
            print(f"[VERBOSE EXCEPTION TRACEBACK]:\n{err_details}")
        finally:
            self.run_btn.configure(state="normal", text="Generate Highlights & Resolve Timeline")
            print("[VERBOSE] Pipeline execution thread finished. UI reset.")

if __name__ == "__main__":
    print("[VERBOSE] Launching application main loop...")
    app = AutoCutApp()
    app.mainloop()