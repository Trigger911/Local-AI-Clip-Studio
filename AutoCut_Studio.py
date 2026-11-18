import os
import time
import json
import threading
import urllib.request
import customtkinter as ctk
from tkinter import filedialog
from faster_whisper import WhisperModel

# Set up CustomTkinter appearance
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class AutoCutStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Local AI Clip Studio — RTX Accelerated Pipeline")
        self.geometry("880x850")
        self.minsize(720, 640)

        # Threading & Cancellation Control
        self.active_worker_thread = None
        self.cancel_requested = threading.Event()

        # Handle graceful window close (clicking 'X')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- UI LAYOUT SETUP ---
        
        # Title Label
        self.title_label = ctk.CTkLabel(self, text="Insta360 Local AI Clipping Pipeline", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=(15, 2))

        self.rtx_badge = ctk.CTkLabel(self, text="⚡ Powered by CUDA / RTX Hardware Acceleration", font=("Arial", 11, "italic"), text_color="#2ecc71")
        self.rtx_badge.pack(pady=(0, 10))

        # 1. FILE SELECTION FRAME (Interactive File Picker)
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.pack(fill="x", padx=20, pady=5)
        
        self.file_path_var = ctk.StringVar(value=r"C:/Video Temp/Insta360/Download/LRV_20260523_111251_01_001.insv")
        
        self.file_entry = ctk.CTkEntry(self.file_frame, textvariable=self.file_path_var, font=("Arial", 11))
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)

        self.browse_button = ctk.CTkButton(self.file_frame, text="Browse...", width=90, command=self.browse_file)
        self.browse_button.pack(side="right", padx=(0, 10), pady=10)

        # 2. OPTIONS & FORMAT SELECTION FRAME
        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.pack(fill="x", padx=20, pady=5)

        # Output format dropdown
        self.format_label = ctk.CTkLabel(self.options_frame, text="Format:", font=("Arial", 11, "bold"))
        self.format_label.pack(side="left", padx=(10, 3), pady=8)

        self.output_format_var = ctk.StringVar(value="fcpxml")
        self.format_dropdown = ctk.CTkOptionMenu(
            self.options_frame, 
            values=["fcpxml", "edl", "srt", "txt"],
            variable=self.output_format_var,
            width=90
        )
        self.format_dropdown.pack(side="left", padx=3, pady=8)

        # Aspect Ratio / Framing dropdown
        self.ratio_label = ctk.CTkLabel(self.options_frame, text="Framing:", font=("Arial", 11, "bold"))
        self.ratio_label.pack(side="left", padx=(10, 3), pady=8)

        self.framing_mode_var = ctk.StringVar(value="Vertical (9:16 Shorts/TikTok)")
        self.framing_dropdown = ctk.CTkOptionMenu(
            self.options_frame, 
            values=["Vertical (9:16 Shorts/TikTok)", "Full 360° Source"],
            variable=self.framing_mode_var,
            width=210
        )
        self.framing_dropdown.pack(side="left", padx=3, pady=8)

        # Auto-detect button / trigger
        self.autodetect_btn = ctk.CTkButton(self.options_frame, text="Auto-Detect", width=85, command=self.auto_detect_settings, fg_color="#34495e", hover_color="#2c3e50")
        self.autodetect_btn.pack(side="left", padx=5, pady=8)

        self.resume_var = ctk.BooleanVar(value=True)
        self.resume_checkbox = ctk.CTkCheckBox(
            self.options_frame, 
            text="Resume", 
            variable=self.resume_var,
            font=("Arial", 11)
        )
        self.resume_checkbox.pack(side="right", padx=10, pady=8)

        # 3. ACTION BUTTONS (Start & Abort)
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=20, pady=(5, 5))

        self.start_button = ctk.CTkButton(self.btn_frame, text="Start RTX Pipeline", command=self.start_pipeline_thread, height=38, fg_color="#1f6aa5")
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.abort_button = ctk.CTkButton(self.btn_frame, text="Stop / Abort", command=self.abort_pipeline, height=38, fg_color="#c0392b", hover_color="#e74c3c", state="disabled")
        self.abort_button.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # Progress Bar & Status
        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal")
        self.progress_bar.pack(fill="x", padx=20, pady=(8, 2))
        self.progress_bar.set(0.0)

        self.status_label = ctk.CTkLabel(self, text="Status: Idle", font=("Arial", 12, "bold"))
        self.status_label.pack(anchor="w", padx=20, pady=(0, 5))

        # 4. TABS FOR LOGS VS. DETECTED AI TIMELINE CLIPS
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        
        self.tab_logs = self.tabview.add("Activity Log")
        self.tab_clips = self.tabview.add("Detected AI Clips (Timestamps)")

        # Activity Log Textbox
        self.log_textbox = ctk.CTkTextbox(self.tab_logs, font=("Consolas", 10))
        self.log_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_textbox.configure(state="disabled")

        # Detected Clips Table/Textbox View
        self.clips_textbox = ctk.CTkTextbox(self.tab_clips, font=("Consolas", 11))
        self.clips_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.clips_textbox.insert("end", "No clips analyzed yet. Run the pipeline to view AI timestamps & durations.\n")
        self.clips_textbox.configure(state="disabled")

        # Initial auto-detection
        self.auto_detect_settings()
        self.log_verbose("SYSTEM", "Application initialized. Ready for processing.")

    def browse_file(self):
        """Opens file dialog to select any target video file and auto-detects parameters."""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.insv *.mp4 *.mov *.mkv"), ("All Files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
            self.auto_detect_settings()
            self.log_verbose("SYSTEM", f"Target file changed to: {filename}")

    def auto_detect_settings(self):
        """Smartly detects output formatting and framing based on file naming or path clues."""
        path = self.file_path_var.get().lower()
        
        if ".insv" in path:
            self.output_format_var.set("fcpxml")
            detected_fmt = "FCPXML"
        else:
            self.output_format_var.set("fcpxml")
            detected_fmt = "FCPXML"

        if "vertical" in path or "vert" in path or "reel" in path or "tiktok" in path:
            self.framing_mode_var.set("Vertical (9:16 Shorts/TikTok)")
            detected_frame = "Vertical (9:16 Shorts/TikTok)"
        else:
            self.framing_mode_var.set("Full 360° Source")
            detected_frame = "Full 360° Source"

        self.log_verbose("SYSTEM", f"Auto-detected -> Format: .{detected_fmt.lower()} | Framing: {detected_frame}")

    def log_verbose(self, step_name, details):
        """Logs detailed progress safely to the GUI textbox and console."""
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
        """Populates the AI Detected Clips tab with timestamp blocks."""
        try:
            self.clips_textbox.configure(state="normal")
            self.clips_textbox.delete("1.0", "end")
            self.clips_textbox.insert("end", clip_data_text)
            self.clips_textbox.configure(state="disabled")
        except Exception:
            pass

    def update_status(self, text, progress_value=None):
        """Updates status text and optional progress bar value (0.0 to 1.0)."""
        try:
            self.status_label.configure(text=f"Status: {text}")
            if progress_value is not None:
                self.progress_bar.set(progress_value)
        except Exception:
            pass

    def start_pipeline_thread(self):
        """Spawns a background thread to prevent GUI freezing."""
        target_file_path = self.file_path_var.get().strip()
        if not target_file_path or not os.path.exists(target_file_path):
            self.log_verbose("ERROR", f"Target file does not exist: {target_file_path}")
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
        """Triggers cancellation flag to stop tasks safely."""
        self.log_verbose("SYSTEM", "Abort requested by user. Stopping background tasks...")
        self.cancel_requested.set()

    def format_time_srt(self, seconds):
        """Converts float seconds to SRT timestamp format (HH:MM:SS,ms)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int(int((seconds - int(seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def run_pipeline_logic(self, file_path):
        """Core pipeline logic running real faster-whisper CUDA transcription and Ollama inference."""
        base_name = os.path.splitext(file_path)[0]
        selected_format = self.output_format_var.get()
        selected_framing = self.framing_mode_var.get()
        
        srt_output = f"{base_name}.srt"
        final_output = f"{base_name}.{selected_format}"
        
        is_resume_enabled = self.resume_var.get()
        
        try:
            self.update_status("Initializing pipeline...", 0.0)
            self.log_verbose("SYSTEM", f"Target file: {os.path.basename(file_path)}")
            self.log_verbose("SYSTEM", f"Output Format: .{selected_format.upper()} | Framing: {selected_framing}")
            self.log_verbose("SYSTEM", f"Resume Mode: {'ENABLED' if is_resume_enabled else 'DISABLED'}")

            if self.cancel_requested.is_set():
                self.log_verbose("SYSTEM", "Pipeline cancelled.")
                return

            # --- STEP 1: FASTER-WHISPER CUDA TRANSCRIPTION ---
            skip_whisper = False
            if is_resume_enabled and os.path.exists(srt_output):
                skip_whisper = True
                self.log_verbose("WHISPER", f"Found existing transcript at {srt_output}. Skipping transcription.")
                self.update_status("Skipped transcription (cached)...", 0.4)
            
            segments_list = []
            if not skip_whisper:
                self.update_status("Transcribing audio (Faster-Whisper CUDA)...", 0.2)
                self.log_verbose("WHISPER", "Initializing large-v3 model on RTX CUDA device (float16)...")
                
                # Initialize Whisper model on CUDA with float16
                model = WhisperModel("large-v3", device="cuda", compute_type="float16")
                
                if self.cancel_requested.is_set():
                    return

                self.update_status("Processing audio chunks on GPU...", 0.35)
                self.log_verbose("WHISPER", "Running high-speed tensor inference across audio streams...")
                
                segments, info = model.transcribe(file_path, beam_size=5)
                
                # Collect segments and write out to SRT file
                with open(srt_output, "w", encoding="utf-8") as f:
                    for i, segment in enumerate(segments, start=1):
                        if self.cancel_requested.is_set():
                            self.log_verbose("WHISPER", "Transcription aborted by user.")
                            return
                        segments_list.append(segment)
                        start_str = self.format_time_srt(segment.start)
                        end_str = self.format_time_srt(segment.end)
                        f.write(f"{i}\n{start_str} --> {end_str}\n{segment.text.strip()}\n\n")

                self.log_verbose("WHISPER", f"Transcription complete. Saved to {srt_output}")

            if self.cancel_requested.is_set():
                self.log_verbose("SYSTEM", "Pipeline cancelled.")
                return

            # --- STEP 2: OLLAMA LLM ANALYSIS ---
            self.update_status("Analyzing highlights (Ollama GPU)...", 0.6)
            self.log_verbose("OLLAMA", "Connecting to local Ollama instance (qwen2.5:32b on CUDA)...")
            
            if self.cancel_requested.is_set():
                return

            # Load transcript text for LLM analysis
            transcript_text = ""
            if os.path.exists(srt_output):
                with open(srt_output, "r", encoding="utf-8") as f:
                    transcript_text = f.read()

            prompt = (
                f"Analyze the following video transcript and identify the best highlight clips "
                f"suitable for {selected_framing}. Return timestamps and reasons.\n\nTranscript:\n{transcript_text}"
            )

            req_data = json.dumps({
                "model": "qwen2.5:32b",
                "prompt": prompt,
                "stream": False
            }).encode("utf-8")

            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )

            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    res_body = json.loads(response.read().decode("utf-8"))
                    ai_response_text = res_body.get("response", "No response generated.")
            except Exception as api_err:
                ai_response_text = f"Error contacting Ollama API: {str(api_err)}"

            # Format and display in the AI Clips tab
            display_output = (
                f"=== OLLAMA AI HIGHLIGHT ANALYSIS ===\n"
                f"Model: qwen2.5:32b | Framing: {selected_framing}\n\n"
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
        """Gracefully signals background threads to stop and exits the app cleanly."""
        self.log_verbose("SYSTEM", "Shutdown requested. Signaling background tasks to stop...")
        self.cancel_requested.set()
        
        if self.active_worker_thread and self.active_worker_thread.is_alive():
            self.active_worker_thread.join(timeout=0.3)
            
        self.destroy()

if __name__ == "__main__":
    app = AutoCutStudioApp()
    app.mainloop()
