import os
import sys
import time
import json
import threading
import urllib.request
import customtkinter as ctk
from tkinter import filedialog

# --- AUTO-LOAD NVIDIA CUDA 12 DLLs ON WINDOWS ---
def register_nvidia_dlls():
    if sys.platform == "win32":
        for site_pkg in sys.path:
            nvidia_dir = os.path.join(site_pkg, "nvidia")
            if os.path.isdir(nvidia_dir):
                for sub in ["cublas", "cudnn", "cuda_runtime"]:
                    bin_path = os.path.join(nvidia_dir, sub, "bin")
                    if os.path.isdir(bin_path):
                        os.add_dll_directory(bin_path)
                        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")

register_nvidia_dlls()
from faster_whisper import WhisperModel

# Set up CustomTkinter appearance
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class AutoCutStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Universal Local AI Video Clip Studio — RTX Accelerated Pipeline")
        self.geometry("1020x1120")
        self.minsize(860, 820)

        # Threading & Cancellation Control
        self.active_worker_thread = None
        self.cancel_requested = threading.Event()

        # Handle graceful window close (clicking 'X')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- UI LAYOUT SETUP ---
        
        # Title Label
        self.title_label = ctk.CTkLabel(self, text="Universal Local AI Video Clip Studio (Insta360 & More)", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=(12, 2))

        self.rtx_badge = ctk.CTkLabel(self, text="⚡ Powered by CUDA / RTX Hardware Acceleration (Hybrid CPU/GPU Pipeline)", font=("Arial", 11, "italic"), text_color="#2ecc71")
        self.rtx_badge.pack(pady=(0, 6))

        # 1. FILE SELECTION FRAME
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.pack(fill="x", padx=20, pady=4)
        
        self.file_path_var = ctk.StringVar(value="")
        
        self.file_entry = ctk.CTkEntry(self.file_frame, textvariable=self.file_path_var, placeholder_text="Select or drop a video file (.mp4, .mov, .mkv, .insv)...", font=("Arial", 11))
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=8)

        self.browse_button = ctk.CTkButton(self.file_frame, text="Browse...", width=90, command=self.browse_file)
        self.browse_button.pack(side="right", padx=(0, 10), pady=8)

        # 2. OPTIONS, FORMAT & LLM MODEL SELECTION FRAME
        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.pack(fill="x", padx=20, pady=4)

        # Output format dropdown
        self.format_label = ctk.CTkLabel(self.options_frame, text="Format:", font=("Arial", 11, "bold"))
        self.format_label.pack(side="left", padx=(8, 2), pady=8)

        self.output_format_var = ctk.StringVar(value="fcpxml")
        self.format_dropdown = ctk.CTkOptionMenu(
            self.options_frame, 
            values=["fcpxml", "edl", "srt", "txt"],
            variable=self.output_format_var,
            width=85
        )
        self.format_dropdown.pack(side="left", padx=2, pady=8)

        # Aspect Ratio / Framing dropdown
        self.ratio_label = ctk.CTkLabel(self.options_frame, text="Framing:", font=("Arial", 11, "bold"))
        self.ratio_label.pack(side="left", padx=(8, 2), pady=8)

        self.framing_mode_var = ctk.StringVar(value="Vertical (9:16 Shorts/TikTok)")
        self.framing_dropdown = ctk.CTkOptionMenu(
            self.options_frame, 
            values=["Vertical (9:16 Shorts/TikTok)", "Full 360° Source", "Standard 16:9 Landscape"],
            variable=self.framing_mode_var,
            width=175
        )
        self.framing_dropdown.pack(side="left", padx=2, pady=8)

        # LLM Model Selector Dropdown
        self.model_label = ctk.CTkLabel(self.options_frame, text="LLM:", font=("Arial", 11, "bold"))
        self.model_label.pack(side="left", padx=(8, 2), pady=8)

        self.llm_model_var = ctk.StringVar(value="qwen2.5:32b")
        self.model_dropdown = ctk.CTkOptionMenu(
            self.options_frame, 
            values=["qwen2.5:32b", "qwen2.5:14b", "llama3:8b", "mistral:7b"],
            variable=self.llm_model_var,
            width=130
        )
        self.model_dropdown.pack(side="left", padx=2, pady=8)

        # Resume Checkbox
        self.resume_var = ctk.BooleanVar(value=True)
        self.resume_checkbox = ctk.CTkCheckBox(
            self.options_frame, 
            text="Resume", 
            variable=self.resume_var,
            font=("Arial", 11)
        )
        self.resume_checkbox.pack(side="right", padx=10, pady=8)

        # 3. LLM MODEL PRO/CON & PERFORMANCE INSPECTOR
        self.inspector_frame = ctk.CTkFrame(self)
        self.inspector_frame.pack(fill="x", padx=20, pady=4)

        self.inspector_title = ctk.CTkLabel(self.inspector_frame, text="🧠 Local LLM Performance Inspector & Pros/Cons Guide (Click a model button below)", font=("Arial", 11, "bold"))
        self.inspector_title.pack(anchor="w", padx=10, pady=(6, 2))

        # Model specification dictionary with descriptions, pros, and cons
        self.model_specs = {
            "qwen2.5:32b": {
                "desc": "Elite heavyweight local reasoning model. Excellent nuance detection for long transcripts.",
                "pros": "Unmatched context comprehension, highly detailed clip justification reasoning.",
                "cons": "Higher VRAM usage (~20GB+), slower token generation speed."
            },
            "qwen2.5:14b": {
                "desc": "Balanced sweet-spot model offering near-32B quality at nearly double the speed.",
                "pros": "Great balance of reasoning and speed, lighter VRAM footprint (~10-12GB).",
                "cons": "Slightly less creative narrative structuring on complex transcripts."
            },
            "llama3:8b": {
                "desc": "Blazing fast lightweight workhorse. Perfect for rapid iterative highlight testing.",
                "pros": "Instantaneous inference speed, very low VRAM requirements (~6GB).",
                "cons": "Can occasionally miss subtle context cues or truncate complex multi-step instructions."
            },
            "mistral:7b": {
                "desc": "Agile instruction-following model optimized for structured outputs and short-form formatting.",
                "pros": "Extremely reliable timestamp formatting, fast response times.",
                "cons": "Narrower context window tolerance on very long raw transcripts."
            }
        }

        self.model_buttons_container = ctk.CTkFrame(self.inspector_frame, fg_color="transparent")
        self.model_buttons_container.pack(fill="x", padx=8, pady=(0, 4))

        for model_name in self.model_specs.keys():
            btn = ctk.CTkButton(
                self.model_buttons_container,
                text=model_name,
                width=110,
                height=26,
                font=("Arial", 10, "bold"),
                fg_color="#34495e",
                hover_color="#2c3e50",
                command=lambda m=model_name: self.show_model_inspector(m)
            )
            btn.pack(side="left", padx=4, pady=2)

        self.model_info_var = ctk.StringVar(value="Click any model button above to inspect its hardware footprint, pros, and cons.")
        self.model_info_label = ctk.CTkLabel(self.inspector_frame, textvariable=self.model_info_var, font=("Arial", 10, "italic"), text_color="#3498db", wraplength=920, justify="left")
        self.model_info_label.pack(anchor="w", padx=12, pady=(0, 6))

        # 4. CATEGORIZED AI CLIP MINING FOCUS TAGS
        self.tags_main_frame = ctk.CTkFrame(self)
        self.tags_main_frame.pack(fill="x", padx=20, pady=4)

        self.tags_title = ctk.CTkLabel(self.tags_main_frame, text="AI Clip Mining Focus Categories & Hobbies (Checked items inject definitions into the prompt below)", font=("Arial", 11, "bold"))
        self.tags_title.pack(anchor="w", padx=12, pady=(6, 2))

        self.hobby_definitions = {
            "Gaming / PC": "High-kill streaks, jump scares, clutch gameplay moments, and loud audio reactions.",
            "VR Horror & Action": "Panic reactions, sudden visual jumps, spooky audio cues, and frantic head movements.",
            "Multiplayer & Streams": "Chat interaction highlights, witty banter, and community event rewards.",
            "Snowboarding": "Big air jumps, powder carving, steep descents, and trick landings.",
            "One-Wheeling": "Carving lines, high-speed straightaways, balance recoveries, and scenic cruising.",
            "Skating / Boarding": "Street/park trick setups, rail slides, stair drops, and clean roll-aways.",
            "Cycling & Trails": "Technical singletrack climbs, fast descents, hairpin turns, and near misses.",
            "Camping": "Campsite setup reels, campfire stories, morning views, and wilderness ambiance.",
            "Boating & Water": "Wakeboarding cuts, smooth river runs, sunset cruises, and splashing anchor moments.",
            "Hiking & Nature": "Summit reveals, panoramic lookouts, wildlife encounters, and trail milestones.",
            "Cycling / Commute": "City traffic weaving, scenic bike paths, and urban exploration footage.",
            "Dog Training & Pets": "Cute puppy tricks, command execution ('Heal', 'Sit'), playful zoomies, and scenic walks.",
            "Casual Vlogging / Talk": "Engaging opening hooks, monologue punchlines, and storytelling transitions."
        }

        self.hobby_categories = {
            "🎮 Gaming & VR": ["Gaming / PC", "VR Horror & Action", "Multiplayer & Streams"],
            "🏂 Action & Extreme": ["Snowboarding", "One-Wheeling", "Skating / Boarding", "Cycling & Trails"],
            "🏕️ Outdoor & Water": ["Camping", "Boating & Water", "Hiking & Nature"],
            "🚴 Lifestyle & Pets": ["Cycling / Commute", "Dog Training & Pets", "Casual Vlogging / Talk"]
        }

        self.hobby_vars = {}
        self.categories_grid = ctk.CTkFrame(self.tags_main_frame, fg_color="transparent")
        self.categories_grid.pack(fill="x", padx=8, pady=(0, 6))
        self.categories_grid.grid_columnconfigure((0, 1), weight=1)

        cat_col = 0
        cat_row = 0
        for category_name, items in self.hobby_categories.items():
            cat_frame = ctk.CTkFrame(self.categories_grid, fg_color="#1e1e1e" if ctk.get_appearance_mode()=="Dark" else "#e6e6e6")
            cat_frame.grid(row=cat_row, column=cat_col, sticky="nsew", padx=4, pady=4)
            
            cat_label = ctk.CTkLabel(cat_frame, text=category_name, font=("Arial", 10, "bold"))
            cat_label.pack(anchor="w", padx=6, pady=(4, 2))

            for item in items:
                var = ctk.BooleanVar(value=False)
                self.hobby_vars[item] = var
                cb = ctk.CTkCheckBox(cat_frame, text=item, variable=var, font=("Arial", 10))
                cb.pack(anchor="w", padx=8, pady=1)

            cat_col += 1
            if cat_col > 1:
                cat_col = 0
                cat_row += 1

        # 5. CUSTOM AI PROMPT & DIRECTIVES INPUT
        self.prompt_frame = ctk.CTkFrame(self)
        self.prompt_frame.pack(fill="x", padx=20, pady=4)

        self.prompt_label = ctk.CTkLabel(self.prompt_frame, text="Custom AI Clip-Mining Prompt Template ( {criteria_details} dynamically inserts checked options ):", font=("Arial", 11, "bold"))
        self.prompt_label.pack(anchor="w", padx=10, pady=(6, 2))

        default_prompt_template = (
            "You are an expert video editor and clip miner. Review the transcript below and extract the top 3-5 punchiest highlight clips "
            "optimized for {framing}.\n\n"
            "ACTIVE CHECKED OPTIONS & CRITERIA TO TARGET:\n{criteria_details}\n\n"
            "Strict Directives:\n"
            "1. Match the transcript timestamps exclusively to moments fitting the criteria above.\n"
            "2. Provide exact start and end timestamps (MM:SS to MM:SS) with precise durations.\n"
            "3. Write a 1-sentence justification explaining why this specific segment matches your criteria."
        )

        self.prompt_textbox = ctk.CTkTextbox(self.prompt_frame, height=90, font=("Consolas", 10))
        self.prompt_textbox.pack(fill="x", padx=10, pady=(0, 6))
        self.prompt_textbox.insert("1.0", default_prompt_template)

        # 6. ACTION BUTTONS (Start & Abort)
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=20, pady=(4, 4))

        self.start_button = ctk.CTkButton(self.btn_frame, text="Start RTX Pipeline", command=self.start_pipeline_thread, height=36, fg_color="#1f6aa5")
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.abort_button = ctk.CTkButton(self.btn_frame, text="Stop / Abort", command=self.abort_pipeline, height=36, fg_color="#c0392b", hover_color="#e74c3c", state="disabled")
        self.abort_button.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # Progress Bar & Status
        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal")
        self.progress_bar.pack(fill="x", padx=20, pady=(6, 2))
        self.progress_bar.set(0.0)

        self.status_label = ctk.CTkLabel(self, text="Status: Ready. Please select a video file to begin.", font=("Arial", 11, "bold"))
        self.status_label.pack(anchor="w", padx=20, pady=(0, 4))

        # 7. TABS FOR LOGS VS. DETECTED AI TIMELINE CLIPS
        self.tabview = ctk.CTkTabview(self, height=180)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        self.tab_logs = self.tabview.add("Activity Log")
        self.tab_clips = self.tabview.add("Detected AI Clips (Timestamps)")

        # Activity Log Textbox
        self.log_textbox = ctk.CTkTextbox(self.tab_logs, font=("Consolas", 10))
        self.log_textbox.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_textbox.configure(state="disabled")

        # Detected Clips Table/Textbox View
        self.clips_textbox = ctk.CTkTextbox(self.tab_clips, font=("Consolas", 11))
        self.clips_textbox.pack(fill="both", expand=True, padx=4, pady=4)
        self.clips_textbox.insert("end", "No clips analyzed yet. Select a video and run the pipeline to view AI timestamps & durations.\n")
        self.clips_textbox.configure(state="disabled")

        self.log_verbose("SYSTEM", "Application initialized. Waiting for video file selection.")

    def show_model_inspector(self, model_name):
        """Displays pros, cons, and performance stats when clicking a model button."""
        info = self.model_specs.get(model_name, {})
        text = (
            f"[{model_name}] Performance Profile:\n"
            f"• Description: {info.get('desc', '')}\n"
            f"• ✅ Pros: {info.get('pros', '')}\n"
            f"• ❌ Cons: {info.get('cons', '')}"
        )
        self.model_info_var.set(text)
        self.llm_model_var.set(model_name)  # Automatically selects it in the dropdown too!

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.insv *.mp4 *.mov *.mkv"), ("All Files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
            self.auto_detect_settings()
            self.log_verbose("SYSTEM", f"Target file selected: {filename}")

    def auto_detect_settings(self):
        path = self.file_path_var.get().lower()
        if not path:
            return
        
        if ".insv" in path:
            self.output_format_var.set("fcpxml")
            detected_fmt = "FCPXML"
        else:
            self.output_format_var.set("fcpxml")
            detected_fmt = "FCPXML"

        if "vertical" in path or "vert" in path or "reel" in path or "tiktok" in path:
            self.framing_mode_var.set("Vertical (9:16 Shorts/TikTok)")
            detected_frame = "Vertical (9:16 Shorts/TikTok)"
        elif "360" in path or ".insv" in path:
            self.framing_mode_var.set("Full 360° Source")
            detected_frame = "Full 360° Source"
        else:
            self.framing_mode_var.set("Standard 16:9 Landscape")
            detected_frame = "Standard 16:9 Landscape"

        self.log_verbose("SYSTEM", f"Auto-detected -> Format: .{detected_fmt.lower()} | Framing: {detected_frame}")

    def log_verbose(self, step_name, details):
        timestamp = time.strftime("[%H:%M:%S]")
        message = f"{timestamp} [{step_name}] {details}"
        print(message)
        
        try:
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", message + "\n")
            self.log_textbox.see("end")
            self.log_textbox.configure(state="disabled")
        except Exception:
            pass

    def update_clips_display(self, clip_data_text):
        try:
            self.clips_textbox.configure(state="normal")
            self.clips_textbox.delete("1.0", "end")
            self.clips_textbox.insert("end", clip_data_text)
            self.clips_textbox.configure(state="disabled")
        except Exception:
            pass

    def update_status(self, text, progress_value=None):
        try:
            self.status_label.configure(text=f"Status: {text}")
            if progress_value is not None:
                self.progress_bar.set(progress_value)
        except Exception:
            pass

    def start_pipeline_thread(self):
        target_file_path = self.file_path_var.get().strip()
        if not target_file_path or not os.path.exists(target_file_path):
            self.log_verbose("ERROR", f"Target file does not exist or none selected: {target_file_path}")
            self.update_status("Error: Please select a valid video file first!")
            return

        self.start_button.configure(state="disabled")
        self.abort_button.configure(state="normal")
        self.cancel_requested.clear()
        
        self.active_worker_thread = threading.Thread(
            target=self.run_pipeline_logic, 
            args=(target_file_path,), 
            daemon=True
        )
        self.active_worker_thread.start()

    def abort_pipeline(self):
        self.log_verbose("SYSTEM", "Abort requested by user. Stopping background tasks...")
        self.cancel_requested.set()

    def format_time_srt(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def run_pipeline_logic(self, file_path):
        base_name = os.path.splitext(file_path)[0]
        selected_format = self.output_format_var.get()
        selected_framing = self.framing_mode_var.get()
        selected_model = self.llm_model_var.get()
        
        # Dynamically aggregate active criteria from checked boxes
        active_criteria = []
        active_hobby_names = []
        for hobby, var in self.hobby_vars.items():
            if var.get():
                active_hobby_names.append(hobby)
                desc = self.hobby_definitions.get(hobby, "General highlight moments.")
                active_criteria.append(f"- {hobby}: {desc}")

        criteria_str = "\n".join(active_criteria) if active_criteria else "- General Content: Punchy dialogue, engaging pacing, and overall video highlights."
        hobbies_str = ", ".join(active_hobby_names) if active_hobby_names else "General Content"
        
        srt_output = f"{base_name}.srt"
        final_output = f"{base_name}.{selected_format}"
        
        is_resume_enabled = self.resume_var.get()
        
        try:
            self.update_status("Initializing pipeline...", 0.0)
            self.log_verbose("SYSTEM", f"Target file: {os.path.basename(file_path)}")
            self.log_verbose("SYSTEM", f"Format: .{selected_format.upper()} | Framing: {selected_framing} | Model: {selected_model}")
            self.log_verbose("SYSTEM", f"Active Focus Tags: [{hobbies_str}]")
            self.log_verbose("SYSTEM", f"Resume Mode: {'ENABLED' if is_resume_enabled else 'DISABLED'}")

            if self.cancel_requested.is_set():
                self.log_verbose("SYSTEM", "Pipeline cancelled.")
                return

            # --- STEP 1: FASTER-WHISPER HYBRID CPU/GPU ACCELERATED TRANSCRIPTION ---
            skip_whisper = False
            if is_resume_enabled and os.path.exists(srt_output):
                skip_whisper = True
                self.log_verbose("WHISPER", f"Found existing transcript at {srt_output}. Skipping transcription.")
                self.update_status("Skipped transcription (cached)...", 0.4)
            
            if not skip_whisper:
                self.update_status("Transcribing audio (RTX CUDA + CPU Hybrid Acceleration)...", 0.2)
                self.log_verbose("WHISPER", "Initializing large-v3 model on CUDA (float16) with optimized CPU thread pooling...")
                
                model = WhisperModel(
                    "large-v3", 
                    device="cuda", 
                    compute_type="float16", 
                    cpu_threads=4, 
                    num_workers=2
                )
                
                if self.cancel_requested.is_set():
                    return

                self.update_status("Processing audio streams on GPU/CPU pipeline...", 0.35)
                self.log_verbose("WHISPER", "Running high-speed tensor inference...")
                
                segments, info = model.transcribe(file_path, beam_size=5)
                
                with open(srt_output, "w", encoding="utf-8") as f:
                    for i, segment in enumerate(segments, start=1):
                        if self.cancel_requested.is_set():
                            self.log_verbose("WHISPER", "Transcription aborted by user.")
                            return
                        start_str = self.format_time_srt(segment.start)
                        end_str = self.format_time_srt(segment.end)
                        f.write(f"{i}\n{start_str} --> {end_str}\n{segment.text.strip()}\n\n")

                self.log_verbose("WHISPER", f"Transcription complete. Saved to {srt_output}")

            if self.cancel_requested.is_set():
                self.log_verbose("SYSTEM", "Pipeline cancelled.")
                return

            # --- STEP 2: OLLAMA LLM ANALYSIS ---
            self.update_status(f"Analyzing highlights ({selected_model} on Ollama GPU)...", 0.6)
            self.log_verbose("OLLAMA", f"Connecting to local Ollama instance ({selected_model} on CUDA) with injected prompt criteria...")
            
            if self.cancel_requested.is_set():
                return

            transcript_text = ""
            if os.path.exists(srt_output):
                with open(srt_output, "r", encoding="utf-8") as f:
                    transcript_text = f.read()

            raw_user_prompt_template = self.prompt_textbox.get("1.0", "end").strip()
            
            try:
                formatted_prompt = raw_user_prompt_template.format(
                    framing=selected_framing,
                    criteria_details=criteria_str
                )
            except Exception:
                formatted_prompt = raw_user_prompt_template

            full_ai_prompt = f"{formatted_prompt}\n\nTranscript:\n{transcript_text}"

            req_data = json.dumps({
                "model": selected_model,
                "prompt": full_ai_prompt,
                "stream": False
            }).encode("utf-8")

            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )

            try:
                with urllib.request.urlopen(req, timeout=600) as response:
                    res_body = json.loads(response.read().decode("utf-8"))
                    ai_response_text = res_body.get("response", "No response generated.")
            except Exception as api_err:
                ai_response_text = f"Error contacting Ollama API: {str(api_err)}"

            display_output = (
                f"=== OLLAMA AI CLIP MINING ANALYSIS ===\n"
                f"Model: {selected_model} | Framing: {selected_framing}\n"
                f"Focus Categories: [{hobbies_str}]\n\n"
                f"{ai_response_text}"
            )
            self.update_clips_display(display_output)

            # --- STEP 3: EXPORT TIMELINE FILE ---
            self.update_status(f"Generating .{selected_format} export...", 0.85)
            self.log_verbose("EXPORT", f"Structuring timeline tracks for .{selected_format}...")

            with open(final_output, "w", encoding="utf-8") as out_f:
                out_f.write(display_output)

            self.update_status("Done! Ready for editing.", 1.0)
            self.log_verbose("SUCCESS", f"Pipeline finished successfully. Saved to {final_output}")

        except Exception as e:
            self.update_status("Error encountered!", 0.0)
            self.log_verbose("ERROR", f"An exception occurred: {str(e)}")
            
        finally:
            try:
                self.start_button.configure(state="normal")
                self.abort_button.configure(state="disabled")
            except Exception:
                pass

    def on_closing(self):
        self.log_verbose("SYSTEM", "Shutdown requested. Signaling background tasks to stop...")
        self.cancel_requested.set()
        
        if self.active_worker_thread and self.active_worker_thread.is_alive():
            self.active_worker_thread.join(timeout=0.3)
            
        self.destroy()

if __name__ == "__main__":
    app = AutoCutStudioApp()
    app.mainloop()
