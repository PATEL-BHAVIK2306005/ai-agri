Jay swaminarayan To all of user
Plant Disease Detection System with Dual AI Engine

![alt text](https://img.shields.io/badge/python-3.8+-blue.svg)


![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)


![alt text](https://img.shields.io/badge/status-active-success.svg)

A comprehensive, user-friendly desktop application built with Python and Tkinter for detecting plant diseases. This system features a powerful dual-engine analysis system, allowing users to choose between a fast, offline local TensorFlow model and an advanced, context-aware Google Gemini Vision API for detailed diagnostics.

The application provides an end-to-end workflow, including automated dataset downloading, integrated model training, and intuitive analysis of plant images.

✨ Key Features

Dual Analysis Engine:

Local Model: Utilizes a pre-trained or custom-trained TensorFlow/Keras model (MobileNetV2) for fast, offline inference.

Cloud AI (Gemini): Integrates with the Google Gemini Vision API to provide expert-level analysis, including detailed symptom descriptions and treatment recommendations.

Integrated Model Training:

Train a deep learning model from scratch using transfer learning.

Real-time GUI updates on training progress, accuracy, and loss.

Visualize performance with automatically generated plots.

Automated Dataset Management:

Download and prepare the recommended PlantVillage dataset directly via the Kaggle API.

Support for downloading custom datasets from a URL.

Interactive & User-Friendly GUI:

Built with Tkinter, featuring a clean, tabbed interface for easy navigation.

Upload images from your local system or capture them directly using a webcam.

All long-running tasks (downloading, training, analysis) are threaded to keep the UI responsive.

Full Model Lifecycle Management:

Save your trained models (.h5) and class mappings (.json).

Load existing models to resume work or perform analysis.

Export models to the TensorFlow Lite (.tflite) format for mobile deployment.

📸 Screenshots
![image](https://github.com/user-attachments/assets/84510a4a-fbb0-43ac-8d72-933461a87557)
![image](https://github.com/user-attachments/assets/394cb142-9c5d-4df5-9b6b-84b1f5d3ae94)

![image](https://github.com/user-attachments/assets/51873322-02df-453f-b875-ba9cd166e8de)
![image](https://github.com/user-attachments/assets/4035eddd-72bf-42f7-b6a7-a98e80863366)






Detection Tab (Gemini Result)	Training Tab	Dataset Management Tab
[Image of the detection tab with a detailed Gemini report]	[Image of the training tab showing plots and logs]	[Image of the dataset tab with download controls]
🛠️ Technology Stack

Backend & ML: Python, TensorFlow, Keras

GUI: Tkinter

Cloud AI: Google Gemini API

Data Handling: Kaggle API, NumPy, OpenCV, Pillow

Plotting: Matplotlib

📋 Prerequisites

Before you begin, ensure you have the following installed and configured:

Python: Version 3.8 or higher.

Pip: Python's package installer.

Kaggle API Key:

You need a Kaggle account.

Download your kaggle.json API token from your Kaggle account page (Account -> API -> Create New API Token).

Place the kaggle.json file in the ~/.kaggle/ directory.

Google Gemini API Key:

Obtain a free API key from the Google AI Studio. This key will be entered directly into the application's GUI.

🚀 Installation & Setup

Follow these steps to get the application running on your local machine.

Clone the Repository

Generated bash
git clone https://github.com/your-username/plant-disease-detector.git
cd plant-disease-detector


Create a Virtual Environment (Recommended)

Generated bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Install Dependencies
A requirements.txt file is recommended. Create one with the following content:

Generated code
tensorflow
opencv-python-headless
Pillow
numpy
matplotlib
requests
kaggle
google-generativeai
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Then, install the packages:

Generated bash
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Run the Application

Generated bash
python main.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Assuming you have saved the code as main.py)

📖 How to Use the Application
1. Dataset Management Tab

This is the best place to start if you don't have a dataset.

Select the PlantVillage Dataset option.

Click Download & Prepare Dataset. The application will use the Kaggle API to download and structure the data into the plant_disease_dataset folder.

2. Model Training Tab

Once you have a dataset, you can train your own local model.

Verify the Dataset Path is correct.

Adjust the hyperparameters (Epochs, Batch Size, etc.) as needed.

Click Start Training. You can monitor the progress in the log and see the final performance plots.

After training, Save your model. It will be saved as a .h5 file along with a .json file for the class names.

3. Disease Detection Tab

This is the main analysis interface.

Choose an Analysis Engine:

Local Model: Select this to use the model you trained or loaded. It's fast and works offline.

Cloud AI (Gemini): Select this for a more detailed analysis. You must first enter your Gemini API Key and click Set.

Upload an Image:

Click Select Image to load a file from your computer.

Click Use Camera to capture a new image.

Analyze:

Click the Analyze Disease button.

The results from your chosen engine will appear in the results panel.

📁 Project Structure
.
├── plant_disease_dataset/    # Directory for downloaded images (created automatically)
│   ├── Apple___Apple_scab/
│   ├── ...
│   └── Tomato___healthy/
├── plant_disease_model.h5    # Default/saved local model file
├── class_names.json          # Class mappings for the saved model
├── main.py                   # The main Python script for the application
├── requirements.txt          # List of Python dependencies
└── README.md                 # This file

📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgments

The PlantVillage Dataset for providing the high-quality images used for training.

The TensorFlow team at Google for creating an incredible deep learning framework.

The Google AI team for the powerful Gemini API.
