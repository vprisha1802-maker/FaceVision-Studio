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
