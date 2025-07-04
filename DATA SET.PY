import tkinter as tk
from tkinter import filedialog, messagebox, ttk, PanedWindow
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
import cv2
import os
import json
import requests
import zipfile
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from pathlib import Path
import re

# Try importing vision and kaggle libraries, handle if not installed
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import kaggle
except (ImportError, OSError):
    kaggle = None


IMG_SIZE = (224, 224)
DEFAULT_MODEL_PATH = "plant_disease_model.h5"
DATASET_DIR = Path("plant_disease_dataset")


class PlantDiseaseDetectorGUI:
    """
    A comprehensive GUI for detecting plant diseases using a dual-engine system:
    a local TensorFlow model and the Google Gemini Vision API. Includes features
    for model training, dataset management, and detailed analysis.
    """
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Plant Disease Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # --- Initialize variables ---
        self.model = None
        self.current_image_path = None
        self.class_names = []
        self.training_thread = None
        self.stop_training_flag = threading.Event()
        self.gemini_api_key_configured = False
        
        # --- Setup core components ---
        self.setup_dataset_directory()
        self.create_widgets()
        self.load_default_model()

    def setup_dataset_directory(self):
        """Create the main dataset directory if it doesn't exist."""
        DATASET_DIR.mkdir(exist_ok=True)

    def create_widgets(self):
        """Create and layout all GUI widgets."""
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        title = tk.Label(main_frame, text="🌱 Plant Disease Detection System", font=('Arial', 24, 'bold'), bg='#f0f0f0', fg='#2d5a27')
        title.pack(pady=(0, 15))

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- Create tabs ---
        self.detection_frame = tk.Frame(self.notebook, bg='#f0f0f0')
        self.training_frame = tk.Frame(self.notebook, bg='#f0f0f0')
        self.dataset_frame = tk.Frame(self.notebook, bg='#f0f0f0')
        
        self.notebook.add(self.detection_frame, text=" 🔎 Disease Detection ")
        self.notebook.add(self.training_frame, text=" 🧠 Model Training ")
        self.notebook.add(self.dataset_frame, text=" 📂 Dataset Management ")

        # --- Populate tabs ---
        self.create_detection_interface()
        self.create_training_interface()
        self.create_dataset_interface()

    def create_detection_interface(self):
        """Create the widgets for the 'Disease Detection' tab."""
        pw = PanedWindow(self.detection_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg='#f0f0f0')
        pw.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Panel: Image and Controls
        left_panel = tk.Frame(pw, bg='#f0f0f0')
        
        # Engine Selection
        engine_frame = tk.LabelFrame(left_panel, text="Analysis Engine", font=('Arial', 12, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        engine_frame.pack(fill=tk.X, pady=(0,10))
        self.analysis_engine_var = tk.StringVar(value="local")
        
        tk.Radiobutton(engine_frame, text="Local Model (TensorFlow)", variable=self.analysis_engine_var, value="local", bg='#f0f0f0').pack(anchor='w')
        gemini_rb = tk.Radiobutton(engine_frame, text="Cloud AI (Gemini Vision)", variable=self.analysis_engine_var, value="gemini", bg='#f0f0f0')
        gemini_rb.pack(anchor='w')

        if genai is None:
            gemini_rb.config(state=tk.DISABLED, text="Cloud AI (Gemini Vision) - 'google-generativeai' not installed")

        # Gemini API Key Config
        gemini_config_frame = tk.Frame(engine_frame, bg='#f0f0f0')
        gemini_config_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(gemini_config_frame, text="Gemini API Key:", bg='#f0f0f0').pack(side=tk.LEFT)
        self.gemini_api_key_var = tk.StringVar()
        tk.Entry(gemini_config_frame, textvariable=self.gemini_api_key_var, show="*", width=30).pack(side=tk.LEFT, padx=5)
        tk.Button(gemini_config_frame, text="Set", command=self.configure_gemini).pack(side=tk.LEFT)

        # Image Upload
        upload_frame = tk.LabelFrame(left_panel, text="Upload Plant Image", font=('Arial', 12, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Button(upload_frame, text="📁 Select Image", command=self.upload_image, font=('Arial', 12), bg='#4CAF50', fg='white', cursor='hand2').pack(fill=tk.X, pady=5)
        tk.Button(upload_frame, text="📷 Use Camera", command=self.use_camera, font=('Arial', 12), bg='#008CBA', fg='white', cursor='hand2').pack(fill=tk.X, pady=5)
        
        self.image_frame = tk.LabelFrame(left_panel, text="Selected Image", font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        self.image_label = tk.Label(self.image_frame, text="No image selected", bg='#e0e0e0', fg='#666')
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        pw.add(left_panel, width=500)

        # Right Panel: Results
        right_panel = tk.Frame(pw, bg='#f0f0f0')
        self.predict_btn = tk.Button(right_panel, text="🔍 Analyze Disease", command=self.predict_disease, font=('Arial', 14, 'bold'), bg='#2196F3', fg='white', cursor='hand2', state=tk.DISABLED)
        self.predict_btn.pack(pady=(0, 15), fill=tk.X)

        results_frame = tk.LabelFrame(right_panel, text="Detection Results", font=('Arial', 12, 'bold'), bg='#f0f0f0')
        results_frame.pack(fill=tk.BOTH, expand=True)
        self.results_text = tk.Text(results_frame, height=10, font=('Arial', 11), wrap=tk.WORD, relief=tk.FLAT)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        
        self.progress = ttk.Progressbar(right_panel, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 5))
        self.status_label = tk.Label(right_panel, text="Ready. Load a model or configure Gemini.", bg='#f0f0f0', fg='#333')
        self.status_label.pack(fill=tk.X)
        pw.add(right_panel)
        
    def create_training_interface(self):
        """Create the widgets for the 'Model Training' tab."""
        pw = PanedWindow(self.training_frame, orient=tk.VERTICAL, sashrelief=tk.RAISED, bg='#f0f0f0')
        pw.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top Panel: Controls and Management
        top_panel = tk.Frame(pw, bg='#f0f0f0')
        
        # Training Controls
        controls_frame = tk.LabelFrame(top_panel, text="Training Parameters", font=('Arial', 12, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(controls_frame, text="Dataset Path:", bg='#f0f0f0').grid(row=0, column=0, sticky='w', pady=2)
        self.dataset_path_var = tk.StringVar(value=str(DATASET_DIR))
        tk.Entry(controls_frame, textvariable=self.dataset_path_var, width=40).grid(row=0, column=1, pady=2)
        tk.Button(controls_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2, padx=5, pady=2)

        param_labels = ["Epochs:", "Batch Size:", "Learning Rate:"]
        self.param_vars = {
            "epochs": tk.StringVar(value="10"),
            "batch_size": tk.StringVar(value="32"),
            "lr": tk.StringVar(value="0.0001")
        }
        for i, (label, key) in enumerate(zip(param_labels, self.param_vars)):
            tk.Label(controls_frame, text=label, bg='#f0f0f0').grid(row=i + 1, column=0, sticky='w', pady=2)
            tk.Entry(controls_frame, textvariable=self.param_vars[key], width=10).grid(row=i + 1, column=1, sticky='w', pady=2)

        self.train_btn = tk.Button(controls_frame, text="🚀 Start Training", command=self.start_training, font=('Arial', 12, 'bold'), bg='#FF9800', fg='white', cursor='hand2')
        self.train_btn.grid(row=4, column=0, pady=10)
        self.stop_btn = tk.Button(controls_frame, text="⏹️ Stop Training", command=self.stop_training, font=('Arial', 12, 'bold'), bg='#F44336', fg='white', cursor='hand2', state=tk.DISABLED)
        self.stop_btn.grid(row=4, column=1, columnspan=2, pady=10)

        # Model Management
        model_frame = tk.LabelFrame(top_panel, text="Model Management", font=('Arial', 12, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        model_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        tk.Button(model_frame, text="💾 Save Trained Model", command=self.save_model).pack(fill=tk.X, pady=4)
        tk.Button(model_frame, text="📂 Load Custom Model", command=self.load_model).pack(fill=tk.X, pady=4)
        tk.Button(model_frame, text="📤 Export to TFLite", command=self.export_model).pack(fill=tk.X, pady=4)

        pw.add(top_panel, height=250)

        # Bottom Panel: Progress, Logs, and Plots
        bottom_panel = tk.Frame(pw, bg='#f0f0f0')
        
        progress_frame = tk.LabelFrame(bottom_panel, text="Training Progress", font=('Arial', 12, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        progress_frame.pack(fill=tk.X)
        self.training_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.training_progress.pack(fill=tk.X, pady=5)
        self.training_status = tk.Label(progress_frame, text="Not training", bg='#f0f0f0')
        self.training_status.pack()

        log_plot_notebook = ttk.Notebook(bottom_panel)
        log_plot_notebook.pack(fill=tk.BOTH, expand=True, pady=(10,0))
        
        # Log Tab
        log_frame = tk.Frame(log_plot_notebook, bg='#f0f0f0')
        self.log_text = tk.Text(log_frame, height=10, font=('Monaco', 9), wrap=tk.WORD, relief=tk.FLAT)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,5), pady=5)
        log_plot_notebook.add(log_frame, text=" 📜 Training Log ")

        # Plot Tab
        self.plot_frame = tk.Frame(log_plot_notebook, bg='white')
        self.fig, self.ax = plt.subplots(1, 2, figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()
        log_plot_notebook.add(self.plot_frame, text=" 📊 Performance Plots ")

        pw.add(bottom_panel)

    def create_dataset_interface(self):
        """Create the widgets for the 'Dataset Management' tab."""
        pw = PanedWindow(self.dataset_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg='#f0f0f0')
        pw.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel: Download Controls
        download_panel = tk.Frame(pw, bg='#f0f0f0', padx=10, pady=10)
        download_frame = tk.LabelFrame(download_panel, text="Download Dataset", font=('Arial', 12, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        download_frame.pack(fill=tk.X)

        info_text = "The recommended dataset is 'PlantVillage' from Kaggle. " \
                    "You must have a 'kaggle.json' API token in your '~/.kaggle' folder to use this feature."
        tk.Label(download_frame, text=info_text, bg='#f0f0f0', wraplength=400, justify=tk.LEFT).pack(anchor='w', pady=(0, 10))
        
        self.dataset_var = tk.StringVar(value="plantvillage")
        kaggle_rb = tk.Radiobutton(download_frame, text="PlantVillage Dataset (from Kaggle)", variable=self.dataset_var, value="plantvillage", bg='#f0f0f0')
        kaggle_rb.pack(anchor='w')
        if kaggle is None:
            kaggle_rb.config(state=tk.DISABLED, text="PlantVillage Dataset (from Kaggle) - 'kaggle' library not installed")

        tk.Radiobutton(download_frame, text="Custom Dataset (from ZIP URL)", variable=self.dataset_var, value="custom", bg='#f0f0f0').pack(anchor='w')
        
        tk.Label(download_frame, text="Custom URL:", bg='#f0f0f0').pack(anchor='w', pady=(10, 0))
        self.custom_url = tk.Entry(download_frame, width=60)
        self.custom_url.pack(fill=tk.X, pady=(0, 10))
        
        self.download_btn = tk.Button(download_frame, text="📥 Download & Prepare Dataset", command=self.download_dataset, font=('Arial', 12, 'bold'), bg='#4CAF50', fg='white', cursor='hand2')
        self.download_btn.pack(pady=10, fill=tk.X)
        pw.add(download_panel, width=500)

        # Right Panel: Dataset Info
        info_panel = tk.Frame(pw, bg='#f0f0f0', padx=10, pady=10)
        info_frame = tk.LabelFrame(info_panel, text="Current Dataset Status", font=('Arial', 12, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        info_frame.pack(fill=tk.BOTH, expand=True)

        self.dataset_info = tk.Text(info_frame, height=15, font=('Monaco', 9), wrap=tk.WORD, relief=tk.FLAT)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.dataset_info.yview)
        self.dataset_info.configure(yscrollcommand=info_scrollbar.set)
        self.dataset_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,5), pady=5)
        self.update_dataset_info()
        pw.add(info_panel)

    # --- UI Update & Helper Methods ---
    def _log_message(self, msg, level="INFO"):
        """Logs a message to the training log text widget."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}][{level}] {msg}\n")
        self.log_text.see(tk.END)

    def _update_status(self, text, in_progress=False):
        """Updates the status label and progress bar."""
        self.status_label.config(text=text)
        if in_progress:
            self.progress.start(10)
        else:
            self.progress.stop()

    # --- Gemini API Methods ---
    def configure_gemini(self):
        """Configures the Gemini API with the provided key."""
        if genai is None:
            messagebox.showerror("Error", "'google-generativeai' library is not installed.")
            return
            
        api_key = self.gemini_api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("API Key Missing", "Please enter your Google Gemini API key.")
            self.gemini_api_key_configured = False
            return
        
        try:
            genai.configure(api_key=api_key)
            # Test the key with a lightweight, non-error-throwing call
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            self.gemini_api_key_configured = True
            messagebox.showinfo("Success", "Gemini API key has been set. The application will validate it on first use.")
        except Exception as e:
            self.gemini_api_key_configured = False
            messagebox.showerror("Configuration Error", f"Failed to configure Gemini API. Please check your key.\nError: {e}")

    # --- Prediction Logic ---
    def predict_disease(self):
        """Determines which prediction engine to use and starts the analysis."""
        if not self.current_image_path:
            messagebox.showwarning("Image Missing", "Please select an image to analyze.")
            return
        
        engine = self.analysis_engine_var.get()
        if engine == 'local' and not self.model:
            messagebox.showwarning("Model Missing", "Local model is not loaded. Please train or load one.")
            return
        if engine == 'gemini' and not self.gemini_api_key_configured:
            messagebox.showwarning("API Key Not Set", "Please set your Gemini API key before using the Cloud AI engine.")
            return

        self.predict_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._predict_worker, args=(engine,), daemon=True).start()

    def _predict_worker(self, engine):
        """Worker function that calls the appropriate prediction engine."""
        try:
            if engine == 'gemini':
                self.root.after(0, lambda: self._update_status("Analyzing with Cloud AI (Gemini)...", in_progress=True))
                self._predict_with_gemini()
            else: # local
                self.root.after(0, lambda: self._update_status("Analyzing with Local Model...", in_progress=True))
                self._predict_with_local_model()
        except Exception as e:
            error_message = f"An error occurred during analysis: {e}"
            self.root.after(0, lambda: messagebox.showerror("Prediction Error", error_message))
        finally:
            self.root.after(0, lambda: self._update_status("Analysis complete. Ready for next image."))
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))

    def _predict_with_gemini(self):
        """Handles the complete Gemini API call and response parsing."""
        try:
            image = Image.open(self.current_image_path)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')

            prompt = """
            You are an expert plant pathologist. Analyze the provided image of a plant leaf.
            Provide your response in a strict JSON format only, with no additional text or markdown.
            The JSON object must have the following keys:
            - "plant_name": (string) The common name of the plant (e.g., "Tomato", "Potato", "Apple"). If unsure, say "Unknown Plant".
            - "is_healthy": (boolean) True if the plant appears healthy, false otherwise.
            - "disease_name": (string) The common name of the disease. If healthy, this should be "Healthy".
            - "confidence_score": (float) Your confidence in the diagnosis, from 0.0 to 1.0.
            - "symptoms_description": (string) A brief, clear description of the symptoms visible in the image that led to your diagnosis.
            - "recommended_treatment": (string) A concise, actionable recommendation for treatment or control. If healthy, suggest a preventative measure.
            """
            
            response = model.generate_content([prompt, image])
            
            # Clean up the response to get pure JSON. It often comes in a markdown block.
            json_text_match = re.search(r'```json\n(.*)\n```', response.text, re.DOTALL)
            if json_text_match:
                cleaned_response = json_text_match.group(1)
            else:
                cleaned_response = response.text

            result_data = json.loads(cleaned_response)
            self.root.after(0, lambda: self._display_gemini_results(result_data))

        except Exception as e:
            response_text = response.text if 'response' in locals() else 'No response received'
            self.root.after(0, lambda: messagebox.showerror("Gemini API Error", f"Failed to get or parse response from Gemini.\nError: {e}\n\nResponse received:\n{response_text}"))


    def _predict_with_local_model(self):
        """Handles prediction using the local TensorFlow model."""
        img = Image.open(self.current_image_path).convert('RGB').resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array, verbose=0)[0]
        top_indices = np.argsort(predictions)[::-1][:3]
        
        results = []
        for idx in top_indices:
            class_name = self.class_names[idx]
            confidence = float(predictions[idx])
            parts = class_name.split('__') if '__' in class_name else class_name.split('___')
            
            plant = parts[0].replace('_', ' ') if len(parts) > 0 else 'Unknown'
            disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Healthy/Unknown'
            
            results.append({'plant': plant, 'disease': disease, 'confidence': confidence})

        self.root.after(0, lambda: self._display_local_results(results))

    # --- Result Display ---
    def _display_gemini_results(self, data):
        """Formats and displays the detailed results from Gemini."""
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.tag_configure("header", font=('Arial', 14, 'bold'), foreground='#2d5a27')
        self.results_text.tag_configure("subheader", font=('Arial', 11, 'bold'), foreground='#005b96')
        self.results_text.tag_configure("normal", font=('Arial', 11))
        
        self.results_text.insert(tk.END, "🌿 Cloud AI Analysis Report 🌿\n\n", "header")
        
        condition = "Healthy" if data.get('is_healthy') else data.get('disease_name', 'Unknown Condition')
        confidence = data.get('confidence_score', 0) * 100
        
        self.results_text.insert(tk.END, f"Plant: ", "subheader")
        self.results_text.insert(tk.END, f"{data.get('plant_name', 'N/A')}\n", "normal")
        self.results_text.insert(tk.END, f"Condition: ", "subheader")
        self.results_text.insert(tk.END, f"{condition} ({confidence:.1f}% confidence)\n\n", "normal")
        
        self.results_text.insert(tk.END, "Symptoms Observed:\n", "subheader")
        self.results_text.insert(tk.END, f"{data.get('symptoms_description', 'N/A')}\n\n", "normal")
        
        self.results_text.insert(tk.END, "Recommended Action:\n", "subheader")
        self.results_text.insert(tk.END, f"{data.get('recommended_treatment', 'N/A')}\n", "normal")

    def _display_local_results(self, results):
        """Formats and displays predictions from the local model."""
        self.results_text.delete(1.0, tk.END)
        self.results_text.tag_configure("header", font=('Arial', 14, 'bold'), foreground='#2d5a27')

        self.results_text.insert(tk.END, "🔬 Local Model Analysis 🔬\n\n", "header")
        
        primary = results[0]
        self.results_text.insert(tk.END, f"Primary Prediction:\n")
        self.results_text.insert(tk.END, f"  - Plant: {primary['plant']}\n")
        self.results_text.insert(tk.END, f"  - Condition: {primary['disease']}\n")
        self.results_text.insert(tk.END, f"  - Confidence: {primary['confidence']:.2%}\n\n")
        
        if len(results) > 1:
            self.results_text.insert(tk.END, "Other Possibilities:\n")
            for res in results[1:]:
                self.results_text.insert(tk.END, f"  - {res['plant']} | {res['disease']} ({res['confidence']:.2%})\n")

    # --- Dataset Management Methods ---
    def update_dataset_info(self):
        """Updates the dataset information display with current status."""
        self.dataset_info.delete(1.0, tk.END)
        self.dataset_info.insert(tk.END, f"Target Dataset Directory: {DATASET_DIR.resolve()}\n")
        self.dataset_info.insert(tk.END, "="*50 + "\n\n")

        if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
            self.dataset_info.insert(tk.END, "Status: No dataset found.\n"
                                             "Please download a dataset using the controls on the left.")
            return

        try:
            class_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir()]
            num_classes = len(class_dirs)
            total_images = sum(len(list(d.glob('*.*'))) for d in class_dirs)

            self.dataset_info.insert(tk.END, f"Status: Dataset found!\n\n")
            self.dataset_info.insert(tk.END, f"Number of classes: {num_classes}\n")
            self.dataset_info.insert(tk.END, f"Total images: {total_images}\n\n")
            self.dataset_info.insert(tk.END, "Classes found:\n")
            for d in class_dirs[:20]: # Show a sample
                self.dataset_info.insert(tk.END, f" - {d.name}\n")
            if len(class_dirs) > 20:
                self.dataset_info.insert(tk.END, " ...and more.\n")

        except Exception as e:
            self.dataset_info.insert(tk.END, f"Status: Error scanning dataset directory.\nError: {e}")

    def download_dataset(self):
        """Starts the dataset download process in a new thread."""
        self.download_btn.config(state=tk.DISABLED, text="Working...")
        if self.dataset_var.get() == "plantvillage":
            if kaggle is None:
                messagebox.showerror("Dependency Error", "Kaggle library not found. Please run 'pip install kaggle' and ensure your API token is set up.")
                self.download_btn.config(state=tk.NORMAL, text="📥 Download & Prepare Dataset")
                return
            threading.Thread(target=self._download_plantvillage_kaggle, daemon=True).start()
        else: # custom
            url = self.custom_url.get().strip()
            if url:
                threading.Thread(target=self._download_custom_zip, args=(url,), daemon=True).start()
            else:
                messagebox.showwarning("Input Error", "Please provide a custom dataset URL.")
                self.download_btn.config(state=tk.NORMAL, text="📥 Download & Prepare Dataset")

    def _download_plantvillage_kaggle(self):
        """Worker to download and extract PlantVillage from Kaggle."""
        self._log_message("Starting PlantVillage dataset download from Kaggle...")
        try:
            if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
                if not messagebox.askyesno("Confirm Overwrite", "Dataset directory is not empty. Overwrite?"):
                    self._log_message("Download cancelled by user.", "WARN")
                    return
                self._log_message(f"Removing existing directory: {DATASET_DIR}")
                shutil.rmtree(DATASET_DIR)
            DATASET_DIR.mkdir()

            self._log_message("Downloading dataset... (this may take a while)")
            kaggle.api.dataset_download_files('sarozpoudel/plant-village-dataset-2023', path=DATASET_DIR, unzip=True, quiet=False)
            self._log_message("Dataset downloaded and unzipped successfully.")

            # The dataset has a nested structure, we need to move the files up.
            source_path = DATASET_DIR / "plantvillage dataset" / "color"
            if not source_path.exists():
                possible_paths = list(DATASET_DIR.rglob('color'))
                if not possible_paths: raise FileNotFoundError("Could not find the 'color' directory in the downloaded dataset.")
                source_path = possible_paths[0]
            
            self._log_message(f"Found image data at: {source_path}")
            self._log_message("Restructuring directories...")
            
            for item in source_path.iterdir():
                if item.is_dir(): shutil.move(str(item), str(DATASET_DIR / item.name))
            
            shutil.rmtree(source_path.parent) # Clean up empty nested structure
            self._log_message("Dataset prepared successfully!")
            self.root.after(0, self.update_dataset_info)
            self.root.after(0, lambda: messagebox.showinfo("Success", "PlantVillage dataset is ready for training!"))

        except Exception as e:
            self._log_message(f"Kaggle download failed: {e}", "ERROR")
            self.root.after(0, lambda: messagebox.showerror("Download Error", f"Failed to download from Kaggle. Ensure 'kaggle.json' is configured correctly.\nError: {e}"))
        finally:
            self.root.after(0, lambda: self.download_btn.config(state=tk.NORMAL, text="📥 Download & Prepare Dataset"))

    def _download_custom_zip(self, url):
        """Worker to download and extract a custom dataset from a ZIP URL."""
        self._log_message(f"Starting custom dataset download from: {url}")
        try:
            if DATASET_DIR.exists(): shutil.rmtree(DATASET_DIR)
            DATASET_DIR.mkdir()
            zip_path = DATASET_DIR / "custom_dataset.zip"

            self._log_message("Downloading...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            
            self._log_message("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(DATASET_DIR)
            
            zip_path.unlink()
            self._log_message("Custom dataset prepared. Please verify the directory structure.")
            self.root.after(0, self.update_dataset_info)
            self.root.after(0, lambda: messagebox.showinfo("Success", "Custom dataset downloaded. Please ensure it has a 'class/images' structure."))

        except Exception as e:
            self._log_message(f"Custom download failed: {e}", "ERROR")
            self.root.after(0, lambda: messagebox.showerror("Download Error", f"Failed to download or extract custom dataset.\nError: {e}"))
        finally:
            self.root.after(0, lambda: self.download_btn.config(state=tk.NORMAL, text="📥 Download & Prepare Dataset"))

    def browse_dataset(self):
        path = filedialog.askdirectory(title="Select Dataset Folder")
        if path:
            self.dataset_path_var.set(path)
            self.update_dataset_info()

    # --- Image Handling ---
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.current_image_path = file_path
            self._display_image(file_path)
            self.predict_btn.config(state=tk.NORMAL)
            self._update_status("Image loaded. Ready for analysis.")

    def use_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access the camera.")
            return

        cam_window = tk.Toplevel(self.root)
        cam_window.title("Camera Capture")
        label = tk.Label(cam_window)
        label.pack()

        def update_frame():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                label.imgtk = imgtk
                label.configure(image=imgtk)
                label.after(10, update_frame)
            else:
                on_close()

        def capture():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                temp_path = "temp_capture.jpg"
                cv2.imwrite(temp_path, frame)
                self.current_image_path = temp_path
                self._display_image(temp_path)
                self.predict_btn.config(state=tk.NORMAL)
                self._update_status("Image captured. Ready for analysis.")
            on_close()

        tk.Button(cam_window, text="📸 Capture", command=capture, font=('Arial', 12, 'bold')).pack(pady=10)
        
        def on_close():
            cap.release()
            cam_window.destroy()

        cam_window.protocol("WM_DELETE_WINDOW", on_close)
        update_frame()

    def _display_image(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((450, 450), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to display image: {e}")
    
    # --- Model Training Methods ---
    def _build_model(self, num_classes):
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
        base_model.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        return Model(inputs=base_model.input, outputs=predictions)

    def start_training(self):
        try:
            epochs = int(self.param_vars['epochs'].get())
            batch_size = int(self.param_vars['batch_size'].get())
            lr = float(self.param_vars['lr'].get())
            dataset_path = Path(self.dataset_path_var.get())

            if not dataset_path.is_dir() or not any(dataset_path.iterdir()):
                messagebox.showerror("Dataset Error", "The specified dataset path is empty or invalid.")
                return

            self.train_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.stop_training_flag.clear()

            self.training_thread = threading.Thread(target=self._train_worker, 
                                                    args=(dataset_path, epochs, batch_size, lr), daemon=True)
            self.training_thread.start()

        except ValueError:
            messagebox.showerror("Input Error", "Epochs, Batch Size, and Learning Rate must be valid numbers.")

    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self._log_message("Stop signal received. Finishing current epoch...", "WARN")
            self.stop_training_flag.set()
            if self.model: self.model.stop_training = True
            self.stop_btn.config(state=tk.DISABLED)

    def _train_worker(self, dataset_path, epochs, batch_size, lr):
        try:
            self._log_message(f"Starting training: Epochs={epochs}, Batch={batch_size}, LR={lr}")
            
            self._log_message("Setting up data generators...")
            datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2,
                                         height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                         horizontal_flip=True, validation_split=0.2, fill_mode='nearest')

            train_generator = datagen.flow_from_directory(dataset_path, target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical', subset='training')
            validation_generator = datagen.flow_from_directory(dataset_path, target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical', subset='validation')
            
            num_classes = len(train_generator.class_indices)
            self.class_names = list(train_generator.class_indices.keys())
            self._log_message(f"Found {num_classes} classes and {train_generator.samples} training images.")

            self._log_message("Building new model...")
            self.model = self._build_model(num_classes)
            self.model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

            gui_callback = TrainingCallback(self)
            self._log_message("Starting model fitting...")
            history = self.model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[gui_callback], verbose=0)

            if not self.stop_training_flag.is_set():
                self._log_message("Training completed successfully!", "SUCCESS")
                self.root.after(0, lambda: self.plot_training_history(history))
                self.root.after(0, lambda: messagebox.showinfo("Training Complete", "Model has been trained and is now active."))
            else:
                self._log_message("Training was stopped by the user.", "WARN")
        
        except Exception as e:
            self._log_message(f"An error occurred during training: {e}", "ERROR")
            self.root.after(0, lambda: messagebox.showerror("Training Error", f"An error occurred: {e}"))
        finally:
            self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.training_progress.config(value=0))
            self.root.after(0, lambda: self.training_status.config(text="Not training"))
            if self.model: self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL if self.current_image_path else tk.DISABLED))

    def plot_training_history(self, history):
        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        self.ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        self.ax[0].set_title('Model Accuracy'); self.ax[0].set_ylabel('Accuracy'); self.ax[0].set_xlabel('Epoch'); self.ax[0].legend(loc='upper left')
        self.ax[1].plot(history.history['loss'], label='Train Loss')
        self.ax[1].plot(history.history['val_loss'], label='Validation Loss')
        self.ax[1].set_title('Model Loss'); self.ax[1].set_ylabel('Loss'); self.ax[1].set_xlabel('Epoch'); self.ax[1].legend(loc='upper left')
        self.fig.tight_layout()
        self.canvas.draw()
        self._log_message("Performance plots updated.")
        
    # --- Model I/O Methods ---
    def save_model(self):
        if not self.model:
            messagebox.showwarning("No Model", "There is no trained model to save.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("HDF5 files", "*.h5")])
        if file_path:
            try:
                self.model.save(file_path)
                class_names_file = Path(file_path).with_suffix('.json')
                with open(class_names_file, 'w') as f: json.dump(self.class_names, f)
                messagebox.showinfo("Success", f"Model saved to {file_path}\nClass names saved to {class_names_file}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save model: {e}")

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("HDF5 files", "*.h5")])
        
        if model_path:
            try:
                self.model = tf.keras.models.load_model(model_path)
                class_names_file = Path(model_path).with_suffix('.json')
                if not class_names_file.exists(): raise FileNotFoundError("class_names.json not found alongside model.")
                with open(class_names_file, 'r') as f: self.class_names = json.load(f)
                
                self._update_status(f"Model '{Path(model_path).name}' loaded. {len(self.class_names)} classes.")
                self.predict_btn.config(state=tk.NORMAL if self.current_image_path else tk.DISABLED)
                messagebox.showinfo("Success", "Model loaded successfully.")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load model: {e}")
                self.model = None

    def load_default_model(self):
        if Path(DEFAULT_MODEL_PATH).exists():
            self._update_status("Loading default model...")
            self.load_model(DEFAULT_MODEL_PATH)
        else:
            self._update_status("Ready. No default model found.")

    def export_model(self):
        if not self.model:
            messagebox.showwarning("No Model", "There is no model to export.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".tflite", filetypes=[("TFLite files", "*.tflite")])
        if file_path:
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
                tflite_model = converter.convert()
                with open(file_path, 'wb') as f: f.write(tflite_model)
                messagebox.showinfo("Success", f"Model exported to TFLite format at {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export model: {e}")


class TrainingCallback(Callback):
    """A custom Keras callback to update the GUI during training."""
    def __init__(self, app_instance):
        super().__init__()
        self.app = app_instance
        self.start_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = datetime.now()
        msg = f"Starting Epoch {epoch + 1}/{self.params['epochs']}"
        self.app.root.after(0, lambda: self.app._log_message(msg, "TRAIN"))
        self.app.root.after(0, lambda: self.app.training_status.config(text=msg))

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = (datetime.now() - self.start_time).total_seconds()
        progress = ((epoch + 1) / self.params['epochs']) * 100
        msg = f"Epoch {epoch + 1} | Loss: {logs['loss']:.4f}, Acc: {logs['accuracy']:.4f} | Val_Loss: {logs['val_loss']:.4f}, Val_Acc: {logs['val_accuracy']:.4f} | Time: {epoch_time:.2f}s"
        
        self.app.root.after(0, lambda: self.app._log_message(msg))
        self.app.root.after(0, lambda: self.app.training_progress.config(value=progress))
        if self.app.stop_training_flag.is_set():
            self.model.stop_training = True


if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseDetectorGUI(root)
    root.mainloop()
