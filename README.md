# 👁️ FaceVision Studio: AI Facial Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://facevision-studio.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-purple)

A state-of-the-art, colorful, interactive Computer Vision & Biometrics web application built with **Streamlit**, **OpenCV**, **Plotly**, **DeepFace**, and **FaceNet**.

---

## 🌐 Live Demo

Experience the full interactive web application directly in your browser:

👉 **[Launch FaceVision Studio](https://facevision-studio.streamlit.app/)**  
🔗 **Direct Link:** `https://facevision-studio.streamlit.app/`

---

## 🌟 Core Features & Algorithms

| Algorithm | Focus Area | What It Does In This App |
| :--- | :--- | :--- |
| 👁️ **Viola-Jones Algorithm** | Classical Detection | Uses Haar feature cascades to locate faces, detect eyes, and identify smiles in real time (~20 ms). |
| 🎯 **Template Matching** | Spatial Cross-Correlation | Slides a facial template patch across the image using normalized correlation (`TM_CCOEFF_NORMED`) and computes 2D heatmaps. |
| 🧠 **DeepFace** | Emotion & Demographics | Detects 7 micro-expressions (Happy, Sad, Angry, Surprised, Fear, Disgust, Neutral), estimated age, gender, and ethnicity with Plotly radar & bar charts. |
| 🧬 **Google FaceNet** | Biometric Metric Learning | Projects faces into 128-dimensional L2-normalized embeddings, computing Cosine Similarity & Euclidean distance for 1-to-1 face verification. |
| 📊 **Multi-Algorithm Matrix** | Comparative Analytics | Side-by-side benchmark comparing classical computer vision against deep neural architectures. |

---

## 🚀 How to Run in Visual Studio Code

### 1️⃣ Open in Visual Studio Code
1. Open Visual Studio Code.
2. Click **File** > **Open Folder...** and select this directory:


### 2️⃣ Run in Terminal (Opens Automatically in Browser!)
Open the built-in terminal in VS Code (`Ctrl + ~` or **Terminal** > **New Terminal**), then run:

```bash
python run.py

📦 Requirements & Installation
If running on a fresh environment or virtual environment, install the dependencies using:
pip install -r requirements.txt

🌐 Deploying to Streamlit Cloud (Free 1-Click Hosting)

You can deploy this web application to the internet for free so anyone can use it online:

Push to GitHub:

Initialize git: git init
Add files: git add .
Commit: git commit -m "Initial commit of FaceVision Studio"
Push to your GitHub repository (e.g., github.com/your-username/facevision-studio).

Deploy on Streamlit Community Cloud:

Go to share.streamlit.io
 and sign in with GitHub.
Click "New app".
Select your repository, branch (main), and set Main file path to app.py.
Click Deploy! Streamlit will automatically install requirements.txt and launch your live public URL!

📂 Project Architecture
face_analytics_app/
│
├── .streamlit/
│   └── config.toml          # Auto-browser opening & colorful theme settings
├── sample_images/           # High-resolution benchmark test faces (Happy, Surprised, Neutral)
├── utils/
│   ├── __init__.py
│   ├── viola_jones.py       # Haar cascades face, eye, smile detector + annotations
│   ├── template_matching.py # Multi-scale normalized cross-correlation & heatmaps
│   ├── deepface_module.py   # Emotion breakdown, demographic & age prediction
│   ├── facenet_module.py    # 128-D embedding extraction, Cosine & Euclidean verification
│   └── sample_data.py       # Synthetic portrait generator for instant testing
│
├── app.py                   # Main Streamlit web application with modern neon glassmorphism UI
├── run.py                   # Auto-browser launcher script for VS Code
├── test_pipeline.py         # Instant validation script for all 4 CV pipelines
├── requirements.txt         # Dependency declarations
└── README.md                # Documentation and deployment guide


🎨 UI & Aesthetics
Neon Glassmorphism Design: Deep navy futuristic background (#0B0F19) with glowing neon purple (#8B5CF6), emerald (#10B981), and cyan (#38BDF8) accents.
Interactive Visualizations: Interactive Plotly polar radar charts, correlation heatmaps, embedding vector fingerprint grids, and custom metrics cards.
Instant Testing: Comes pre-packaged with 3 sample benchmark portraits, custom image upload, and webcam snapshot capabilities.

---

### How to Update It on GitHub:

1. Open your repository on **[github.com](https://github.com)**.
2. Click on **`README.md`**.
3. Click the **✏️ (Pencil icon)** to edit.
4. Replace the text with the block above.
5. Click **"Commit changes"**!
