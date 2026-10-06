"""
FaceVision Studio - Next-Gen Facial Intelligence Suite
A comprehensive web platform integrating:
- Template Matching (cv2.matchTemplate)
- Viola-Jones Algorithm (Haar Feature Cascades)
- DeepFace (Emotion, Age, Gender, Race Analysis)
- FaceNet (128-D Embedding & Face Verification)
"""

import os
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Import our custom CV modules
from utils.viola_jones import run_viola_jones
from utils.template_matching import run_template_matching, MATCH_METHODS
from utils.deepface_module import analyze_face_deepface, EMOTION_EMOJIS, EMOTION_COLORS, DEEPFACE_AVAILABLE
from utils.facenet_module import extract_facenet_embedding, verify_faces
from utils.sample_data import get_available_samples

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="FaceVision Studio | AI Face Intelligence",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling: Modern Dark Neon & Glassmorphism Theme
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background gradient styling */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(26, 31, 56, 0.95) 0%, rgba(11, 15, 25, 1) 90%);
    color: #F8FAFC;
}

/* Hero Header styling */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 30px 35px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px rgba(139, 92, 246, 0.15);
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF6B6B 0%, #A78BFA 45%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94A3B8;
    margin-bottom: 18px;
    line-height: 1.5;
}

.pill-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 10px;
}

.pill {
    font-size: 0.78rem;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.pill-viola { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
.pill-template { background: rgba(244, 63, 94, 0.15); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.3); }
.pill-deepface { background: rgba(139, 92, 246, 0.15); color: #C084FC; border: 1px solid rgba(139, 92, 246, 0.3); }
.pill-facenet { background: rgba(14, 165, 233, 0.15); color: #38BDF8; border: 1px solid rgba(14, 165, 233, 0.3); }

/* Glassmorphic Metric Cards */
.metric-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(167, 139, 250, 0.4);
}

.metric-val {
    font-size: 2.0rem;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}

.metric-lbl {
    font-size: 0.82rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}

/* Emotion Feature Card */
.emotion-badge {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    margin-bottom: 20px;
}

.emotion-icon {
    font-size: 3.5rem;
    margin-bottom: 8px;
    display: inline-block;
}

.emotion-text {
    font-size: 1.8rem;
    font-weight: 700;
    color: #F8FAFC;
    text-transform: capitalize;
}

/* Section Container Cards */
.card-box {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
}

/* Custom tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(15, 23, 42, 0.6);
    padding: 8px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    color: #94A3B8;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.35);
}

/* Code block highlight */
code {
    font-family: 'JetBrains Mono', monospace !important;
    color: #38BDF8 !important;
    background: rgba(15, 23, 42, 0.8) !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions: Image Conversion
# ---------------------------------------------------------
def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR numpy array."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(bgr_image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR to RGB for Streamlit display."""
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------
# Sidebar: Input Sources & Parameter Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ **Vision Control Center**")
    st.caption("Upload images or select presets to run real-time facial analytics.")

    input_mode = st.radio(
        "Image Input Source",
        ["📁 Upload Image", "🎨 Sample Presets", "📷 Take Snapshot"],
        index=1,
        help="Choose whether to upload your photo or test instantly with high-res presets."
    )

    current_image_bgr = None
    samples_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    available_samples = get_available_samples(samples_dir)

    if input_mode == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Choose a Portrait or Face Photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="High-quality front-facing photos work best."
        )
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            current_image_bgr = pil_to_bgr(pil_img)
        else:
            st.info("💡 Upload an image to start, or switch to 'Sample Presets'.")

    elif input_mode == "🎨 Sample Presets":
        sample_names = [os.path.basename(p) for p in available_samples]
        chosen_sample_name = st.selectbox(
            "Select Benchmark Sample Face",
            sample_names,
            index=0
        )
        chosen_path = os.path.join(samples_dir, chosen_sample_name)
        if os.path.exists(chosen_path):
            current_image_bgr = cv2.imread(chosen_path)

    elif input_mode == "📷 Take Snapshot":
        cam_photo = st.camera_input("Take a photo with your webcam")
        if cam_photo is not None:
            pil_img = Image.open(cam_photo)
            current_image_bgr = pil_to_bgr(pil_img)

    st.markdown("---")
    st.markdown("### ⚙️ **Algorithm Settings**")

    # Viola-Jones Sliders
    with st.expander("👁️ Viola-Jones Parameters", expanded=False):
        vj_scale = st.slider("Scale Factor (Pyramid step)", 1.05, 1.40, 1.15, 0.05,
                             help="How much image size is reduced at each image scale.")
        vj_neighbors = st.slider("Min Neighbors", 1, 12, 5, 1,
                                 help="Higher values reduce false positives.")
        vj_detect_eyes = st.checkbox("Detect Eyes", value=True)
        vj_detect_smile = st.checkbox("Detect Smile", value=True)

    # Template Matching Settings
    with st.expander("🎯 Template Matching Parameters", expanded=False):
        tm_method_name = st.selectbox(
            "Matching Metric",
            list(MATCH_METHODS.keys()),
            index=0
        )
        tm_multiscale = st.checkbox("Enable Multi-Scale Search", value=True,
                                    help="Searches at multiple scales to accommodate sizing differences.")

    # FaceNet Settings
    with st.expander("🧬 FaceNet Parameters", expanded=False):
        fn_threshold = st.slider("Cosine Similarity Threshold", 0.40, 0.90, 0.65, 0.05,
                                 help="Threshold above which two faces are declared identical.")

    st.markdown("---")
    # System Status Indicator
    st.markdown("### 💡 **Engine Status**")
    if DEEPFACE_AVAILABLE:
        st.success("🟢 **DeepFace Framework:** Native Active")
    else:
        st.markdown(
            """
            <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 10px; font-size: 0.82rem;">
                <b style="color: #38BDF8;">⚡ OpenCV + AI Heuristic Engine Active</b><br/>
                Runs fast & offline without extra GPU weights. To enable deep neural weights anytime: <code>pip install deepface tf-keras</code>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">FaceVision Studio</div>
        <div class="hero-subtitle">
            An interactive computer vision platform exploring classical detection, spatial correlation, 
            and deep metric learning for facial perception and emotion analytics.
        </div>
        <div class="pill-container">
            <span class="pill pill-viola">✨ Viola-Jones Cascades</span>
            <span class="pill pill-template">🎯 Template Matching</span>
            <span class="pill pill-deepface">🧠 DeepFace Emotions</span>
            <span class="pill pill-facenet">🧬 FaceNet 128-D Embeddings</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Guard: If no image is available
if current_image_bgr is None:
    st.warning("👈 Please upload an image or choose a Sample Preset from the left sidebar to begin analysis.")
    st.stop()


# ---------------------------------------------------------
# Global Pre-Processing: Run Initial Viola-Jones to get Face ROIs
# ---------------------------------------------------------
vj_results = run_viola_jones(
    current_image_bgr,
    scale_factor=vj_scale,
    min_neighbors=vj_neighbors,
    detect_eyes=vj_detect_eyes,
    detect_smile=vj_detect_smile
)

# Extract primary face crop (if detected) or center crop fallback
if vj_results["faces_count"] > 0 and len(vj_results["face_crops"]) > 0:
    primary_face_crop = vj_results["face_crops"][0]
else:
    # Use center 60% crop as fallback face ROI
    h, w = current_image_bgr.shape[:2]
    ymin, ymax = int(h * 0.15), int(h * 0.85)
    xmin, xmax = int(w * 0.15), int(w * 0.85)
    primary_face_crop = current_image_bgr[ymin:ymax, xmin:xmax]

# Run DeepFace / Emotion analysis on primary face
deepface_results = analyze_face_deepface(primary_face_crop)

# Run FaceNet embedding extraction on primary face
facenet_results = extract_facenet_embedding(primary_face_crop)


# ---------------------------------------------------------
# Top Highlights Metrics Bar
# ---------------------------------------------------------
mcol1, mcol2, mcol3, mcol4 = st.columns(4)

with mcol1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #34D399;">{vj_results['faces_count']}</div>
            <div class="metric-lbl">Faces Detected</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with mcol2:
    dom_em = deepface_results['dominant_emotion'].capitalize()
    emoji = EMOTION_EMOJIS.get(deepface_results['dominant_emotion'], "✨")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #A78BFA;">{emoji} {dom_em}</div>
            <div class="metric-lbl">Dominant Emotion</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with mcol3:
    age_est = deepface_results['estimated_age']
    gender_est = deepface_results['dominant_gender']
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #38BDF8;">~{age_est} <span style="font-size: 1.1rem;">yrs</span></div>
            <div class="metric-lbl">Est. Age ({gender_est})</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with mcol4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #FB7185;">128-D</div>
            <div class="metric-lbl">FaceNet Vector Dimension</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("") # Spacer


# ---------------------------------------------------------
# Main Tabs: The 4 Core Algorithms + Comprehensive Comparison
# ---------------------------------------------------------
tab_vj, tab_tm, tab_df, tab_fn, tab_compare = st.tabs([
    "👁️ Viola-Jones Algorithm",
    "🎯 Template Matching",
    "🧠 DeepFace & Emotion Analysis",
    "🧬 FaceNet & Biometric Verification",
    "📊 Multi-Algorithm Benchmark"
])


# =========================================================
# TAB 1: VIOLA-JONES ALGORITHM
# =========================================================
with tab_vj:
    st.markdown("### 👁️ **Viola-Jones Object Detection Framework (Haar Cascades)**")
    st.caption("Pioneered by Paul Viola and Michael Jones in 2001: The first real-time capable facial detection algorithm.")
    
    col_vj_img1, col_vj_img2 = st.columns([1, 1])

    with col_vj_img1:
        st.markdown("##### 📷 **Original Image**")
        st.image(bgr_to_rgb(current_image_bgr), use_container_width=True)

    with col_vj_img2:
        st.markdown("##### 🎯 **Haar Cascades Detection Result**")
        st.image(bgr_to_rgb(vj_results["annotated_bgr"]), use_container_width=True)

    st.markdown("---")
    
    # Inspection of detected crops & breakdown
    c_crops, c_stats = st.columns([1, 1.2])
    
    with c_crops:
        st.markdown("##### 🖼️ **Extracted Face Region(s)**")
        if vj_results["faces_count"] > 0:
            crop_cols = st.columns(min(3, vj_results["faces_count"]))
            for idx, crop in enumerate(vj_results["face_crops"]):
                col_idx = idx % 3
                with crop_cols[col_idx]:
                    st.image(bgr_to_rgb(crop), caption=f"Face #{idx + 1}", use_container_width=True)
        else:
            st.info("No distinct face detected with current sensitivity. Try decreasing the 'Min Neighbors' slider in the left sidebar.")

    with c_stats:
        st.markdown("##### ⚡ **Viola-Jones Execution Metrics**")
        df_vj = pd.DataFrame([
            {"Metric": "Algorithm", "Value": "Viola-Jones (Haar Cascades)"},
            {"Metric": "Execution Latency", "Value": f"{vj_results['execution_time_ms']} ms"},
            {"Metric": "Faces Located", "Value": str(vj_results['faces_count'])},
            {"Metric": "Scale Factor (Pyramid)", "Value": f"{vj_results['scale_factor']}"},
            {"Metric": "Min Neighbors Filter", "Value": str(vj_results['min_neighbors'])},
        ])
        st.dataframe(df_vj, use_container_width=True, hide_index=True)

    # Educational Deep Dive
    with st.expander("📚 **How the Viola-Jones Algorithm Works (Under The Hood)**", expanded=False):
        st.markdown(
            """
            The Viola-Jones detector is built on **four revolutionary architectural pillars**:
            
            1. **Haar-like Features:** Digital rectangle features (edge, line, and four-rectangle features) that calculate the difference between the sum of pixel intensities in adjacent white and black rectangular regions.
            2. **Integral Image Representation:** A spatial summed-area table allowing the sum of pixels over any rectangular area to be calculated in **$O(1)$ constant time** using just 4 array lookups:
               $$\\text{Sum}(D) = I(x_2, y_2) - I(x_1, y_2) - I(x_2, y_1) + I(x_1, y_1)$$
            3. **AdaBoost Machine Learning:** Boosts an ensemble of weak classifiers into a single strong classifier, selecting the most informative features out of 160,000+ possibilities.
            4. **Attentional Cascade Architecture:** Evaluates simple stages first. Non-face background regions are rejected within the first 1-2 stages (~95% rejection rate), allowing real-time 30+ FPS operation!
            """
        )


# =========================================================
# TAB 2: TEMPLATE MATCHING
# =========================================================
with tab_tm:
    st.markdown("### 🎯 **Template Matching & Spatial Cross-Correlation**")
    st.caption("Locates a template image patch within a larger scene by sliding and computing mathematical similarity.")

    tm_col_left, tm_col_right = st.columns([1, 1.2])

    with tm_col_left:
        st.markdown("##### 1️⃣ **Select or Provide Search Template**")
        template_choice = st.radio(
            "Template Source",
            ["✨ Auto-Detected Face Crop", "📂 Upload Custom Template Image"],
            index=0
        )

        template_bgr = None
        if template_choice == "✨ Auto-Detected Face Crop":
            template_bgr = primary_face_crop
            st.image(bgr_to_rgb(template_bgr), caption="Active Search Template (Detected Face)", width=180)
        else:
            tpl_upload = st.file_uploader("Upload Template Image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="tm_upload")
            if tpl_upload is not None:
                tpl_pil = Image.open(tpl_upload)
                template_bgr = pil_to_bgr(tpl_pil)
                st.image(bgr_to_rgb(template_bgr), caption="Custom Search Template", width=180)
            else:
                template_bgr = primary_face_crop
                st.info("Upload a template image, or fallback will use the detected face.")

    with tm_col_right:
        st.markdown("##### 2️⃣ **Matching Result & Confidence**")
        if template_bgr is not None:
            tm_res = run_template_matching(
                target_bgr=current_image_bgr,
                template_bgr=template_bgr,
                method_name=tm_method_name,
                multi_scale=tm_multiscale
            )

            # Score meter
            conf = tm_res["confidence_percent"]
            score_color = "#10B981" if conf > 75 else ("#F59E0B" if conf > 45 else "#EF4444")
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.6); padding: 14px 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; color: #94A3B8;">Match Confidence:</span>
                        <span style="font-size: 1.5rem; font-weight: 800; color: {score_color};">{conf:.1f}%</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">
                        Method: <code>{tm_res['method_name']}</code> | Optimal Scale: <b>{tm_res['best_scale']}x</b> | Time: <b>{tm_res['execution_time_ms']} ms</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            tm_res_col1, tm_res_col2 = st.columns(2)
            with tm_res_col1:
                st.image(bgr_to_rgb(tm_res["annotated_bgr"]), caption="Localised Template in Target", use_container_width=True)
            with tm_res_col2:
                st.image(bgr_to_rgb(tm_res["heatmap_color_bgr"]), caption="Correlation Response Heatmap (Inferno)", use_container_width=True)

    st.markdown("---")
    with st.expander("🔬 **Mathematical Principles of Template Matching**"):
        st.markdown(
            """
            In normalized correlation coefficient matching (`TM_CCOEFF_NORMED`), the correlation score $R(x,y)$ at coordinate $(x,y)$ is given by:
            $$R(x,y) = \\frac{\\sum_{x',y'} (T'(x',y') \\cdot I'(x+x', y+y'))}{\\sqrt{\\sum_{x',y'} T'(x',y')^2 \\cdot \\sum_{x',y'} I'(x+x', y+y')^2}}$$
            where:
            - $T'(x',y') = T(x',y') - \\bar{T}$ (zero-mean template patch)
            - $I'(x+x', y+y') = I(x+x', y+y') - \\bar{I}_{x,y}$ (zero-mean image window)
            - A value of **$+1.0$** indicates an absolute pixel correlation match, **$0.0$** indicates no correlation, and **$-1.0$** indicates inverse correlation.
            """
        )


# =========================================================
# TAB 3: DEEPFACE EMOTION & ATTRIBUTE RECOGNITION
# =========================================================
with tab_df:
    st.markdown("### 🧠 **DeepFace Multi-Attribute & Emotion Recognition**")
    st.caption("Deep representation learning assessing subtle micro-expressions, age brackets, gender, and demographics.")

    df_col_left, df_col_mid, df_col_right = st.columns([1, 1.2, 1.2])

    with df_col_left:
        st.markdown("##### 🎭 **Dominant Expression**")
        dom_em = deepface_results["dominant_emotion"]
        dom_color = EMOTION_COLORS.get(dom_em, "#8B5CF6")
        dom_emoji = EMOTION_EMOJIS.get(dom_em, "✨")

        st.markdown(
            f"""
            <div class="emotion-badge" style="border-color: {dom_color};">
                <div class="emotion-icon">{dom_emoji}</div>
                <div class="emotion-text" style="color: {dom_color};">{dom_em.upper()}</div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px;">Dominant Emotion</div>
                <div style="margin-top: 15px; font-weight: 700; font-size: 1.3rem; color: #F8FAFC;">
                    {deepface_results['emotion_scores'].get(dom_em, 0.0):.1f}%
                </div>
                <div style="font-size: 0.75rem; color: #64748B;">Prediction Confidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card-box">
                <div style="font-weight: 600; color: #94A3B8; margin-bottom: 8px;">Demographic Estimates</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span>Estimated Age:</span> <b>{deepface_results['estimated_age']} years</b>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span>Predicted Gender:</span> <b>{deepface_results['dominant_gender']}</b>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Ethnicity / Group:</span> <b>{deepface_results['dominant_race'].capitalize()}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with df_col_mid:
        st.markdown("##### 📊 **Emotion Probability Distribution**")
        em_df = pd.DataFrame(list(deepface_results["emotion_scores"].items()), columns=["Emotion", "Probability"])
        em_df["Emotion"] = em_df["Emotion"].str.capitalize()
        em_df = em_df.sort_values(by="Probability", ascending=True)

        fig_bar = px.bar(
            em_df,
            x="Probability",
            y="Emotion",
            orientation="h",
            color="Emotion",
            color_discrete_map={k.capitalize(): v for k, v in EMOTION_COLORS.items()},
            text="Probability",
        )
        fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=10, r=40, t=10, b=10),
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", range=[0, 105]),
            yaxis=dict(showgrid=False),
            height=320
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with df_col_right:
        st.markdown("##### 🕸️ **Facial Affect Radar Chart**")
        # Radar chart for all emotions
        radar_categories = [k.capitalize() for k in deepface_results["emotion_scores"].keys()]
        radar_values = list(deepface_results["emotion_scores"].values())
        # Close radar loop
        radar_categories.append(radar_categories[0])
        radar_values.append(radar_values[0])

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_categories,
            fill='toself',
            fillcolor='rgba(139, 92, 246, 0.35)',
            line=dict(color='#8B5CF6', width=2),
            marker=dict(size=5, color='#C084FC')
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(radar_values) + 10], color="#64748B"),
                bgcolor="rgba(0,0,0,0)"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8"),
            margin=dict(l=30, r=30, t=20, b=20),
            height=320,
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown(f"**Analytics Engine:** `{deepface_results['engine']}` | **Inference Time:** `{deepface_results.get('execution_time_ms', 0.0)} ms`")


# =========================================================
# TAB 4: FACENET EMBEDDINGS & FACE VERIFICATION
# =========================================================
with tab_fn:
    st.markdown("### 🧬 **Google FaceNet: 128-D Euclidean Metric Learning & Verification**")
    st.caption("Maps facial geometry directly onto a hypersphere where squared L2 distance corresponds directly to face similarity.")

    fn_col1, fn_col2 = st.columns([1, 1.2])

    with fn_col1:
        st.markdown("##### 👤 **Face 1 (Reference Profile)**")
        st.image(bgr_to_rgb(primary_face_crop), caption="Face 1 (Active Subject)", width=220)

    with fn_col2:
        st.markdown("##### 👥 **Face 2 (Verification Candidate)**")
        fn_face2_mode = st.radio(
            "Select Second Face to Compare:",
            ["Preset Alternate Sample", "Upload Second Image", "Use Same Face (Self-Test)"],
            index=0,
            horizontal=True
        )

        face2_bgr = None
        if fn_face2_mode == "Use Same Face (Self-Test)":
            face2_bgr = primary_face_crop
        elif fn_face2_mode == "Upload Second Image":
            f2_upload = st.file_uploader("Upload Second Face Photo", type=["jpg", "jpeg", "png"], key="fn_upload2")
            if f2_upload is not None:
                face2_bgr = pil_to_bgr(Image.open(f2_upload))
            else:
                face2_bgr = cv2.flip(primary_face_crop, 1) # Horizontal mirror as alternative test
        else:
            # Load alternate sample from presets
            alt_sample = available_samples[1] if len(available_samples) > 1 else available_samples[0]
            alt_img = cv2.imread(alt_sample)
            alt_vj = run_viola_jones(alt_img)
            face2_bgr = alt_vj["face_crops"][0] if alt_vj["faces_count"] > 0 else alt_img

        if face2_bgr is not None:
            st.image(bgr_to_rgb(face2_bgr), caption="Face 2 (Comparison Candidate)", width=220)

    # Extract Embedding 2 and perform verification
    if face2_bgr is not None:
        emb2_data = extract_facenet_embedding(face2_bgr)
        verify_res = verify_faces(
            facenet_results["embedding"],
            emb2_data["embedding"],
            cosine_threshold=fn_threshold
        )

        st.markdown("---")
        st.markdown("##### ⚖️ **FaceNet Biometric Verification Decision**")

        # Decision Alert Card
        if verify_res["is_match"]:
            match_html = f"""
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 16px; padding: 20px; display: flex; align-items: center; gap: 20px;">
                <span style="font-size: 2.8rem;">✅</span>
                <div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #34D399;">{verify_res['verdict']}</div>
                    <div style="color: #A7F3D0; font-size: 0.9rem;">Cosine Similarity: <b>{verify_res['cosine_similarity']:.4f}</b> (Threshold: {verify_res['cosine_threshold']:.2f}) | Euclidean L2 Distance: <b>{verify_res['euclidean_distance']:.4f}</b></div>
                </div>
            </div>
            """
        else:
            match_html = f"""
            <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 16px; padding: 20px; display: flex; align-items: center; gap: 20px;">
                <span style="font-size: 2.8rem;">❌</span>
                <div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #F87171;">{verify_res['verdict']}</div>
                    <div style="color: #FECACA; font-size: 0.9rem;">Cosine Similarity: <b>{verify_res['cosine_similarity']:.4f}</b> (Threshold: {verify_res['cosine_threshold']:.2f}) | Euclidean L2 Distance: <b>{verify_res['euclidean_distance']:.4f}</b></div>
                </div>
            </div>
            """
        st.markdown(match_html, unsafe_allow_html=True)

        st.markdown("##### 🧬 **128-Dimensional Embedding Fingerprint Heatmap**")
        st.caption("Visualizing the 128 numerical weights of Face 1's embedding vector reshaped into a 8×16 matrix:")

        emb1_grid = facenet_results["embedding"][:128].reshape(8, 16)
        fig_heat = px.imshow(
            emb1_grid,
            color_continuous_scale="Viridis",
            labels=dict(x="Feature Block X", y="Feature Block Y", color="Vector Weight"),
            aspect="auto"
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8"),
            height=260,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Dimension line profile comparison
        st.markdown("##### 📈 **Dimensional Overlay: Face 1 vs Face 2**")
        line_df = pd.DataFrame({
            "Dimension": list(range(1, 129)),
            "Face 1 Embedding": facenet_results["embedding"][:128],
            "Face 2 Embedding": emb2_data["embedding"][:128],
        })
        fig_line = px.line(
            line_df,
            x="Dimension",
            y=["Face 1 Embedding", "Face 2 Embedding"],
            color_discrete_sequence=["#38BDF8", "#F43F5E"]
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            height=260,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with st.expander("🔬 **Deep Dive: FaceNet Triplet Loss Formulation**"):
        st.markdown(
            """
            FaceNet replaces softmax classifiers with a direct metric learning loss termed **Triplet Loss**:
            $$\\mathcal{L} = \\sum_{i}^{N} \\left[ \\| f(x_i^a) - f(x_i^p) \\|_2^2 - \\| f(x_i^a) - f(x_i^n) \\|_2^2 + \\alpha \\right]_+$$
            Where:
            - $x^a$: **Anchor** face image of person $X$.
            - $x^p$: **Positive** face image (same person $X$ under different pose/lighting).
            - $x^n$: **Negative** face image (different person $Y$).
            - $\\alpha$: Enforcement margin separating clusters.
            
            This guarantees that all faces of the same person cluster tightly within a hyper-sphere of radius $< 1.0$, while different people are separated by large margins!
            """
        )


# =========================================================
# TAB 5: MULTI-ALGORITHM BENCHMARK & COMPARISON
# =========================================================
with tab_compare:
    st.markdown("### 📊 **Comprehensive Algorithm Benchmark & Comparison**")
    st.caption("How Classical Computer Vision compares directly against Modern Deep Metric Learning:")

    comparison_data = [
        {
            "Algorithm": "Viola-Jones (Haar Cascades)",
            "Paradigm": "Classical Machine Learning (AdaBoost + Haar Features)",
            "Primary Function": "Object & Face Bounding Box Localization",
            "Latency": "~10 - 30 ms (Very Fast, CPU friendly)",
            "Lighting / Pose Invariance": "Low (Requires front-facing lighting)",
            "Strengths": "Blazing fast, zero GPU requirement, tiny memory footprint",
            "Weaknesses": "Rigid, fails on tilted/profile faces, no identity recognition"
        },
        {
            "Algorithm": "Template Matching",
            "Paradigm": "Pixel Cross-Correlation & Signal Processing",
            "Primary Function": "Exact Image Patch & Texture Matching",
            "Latency": "~5 - 50 ms (Depends on scale pyramid)",
            "Lighting / Pose Invariance": "Very Low (Extremely sensitive to rotation & lighting)",
            "Strengths": "Direct pixel tracking, mathematically transparent, no training needed",
            "Weaknesses": "Brittle to expression changes, 3D rotations, and occlusions"
        },
        {
            "Algorithm": "DeepFace",
            "Paradigm": "Deep Convolutional Neural Networks (VGG-Face / ResNet)",
            "Primary Function": "Multi-Task Demographic & Emotion Attribute Classification",
            "Latency": "~100 - 300 ms (CPU/GPU)",
            "Lighting / Pose Invariance": "High (Learned robust feature representations)",
            "Strengths": "Understands emotions, expressions, age, and soft biometrics",
            "Weaknesses": "Requires larger model weights and higher computation"
        },
        {
            "Algorithm": "FaceNet",
            "Paradigm": "Deep Metric Learning with Triplet Loss",
            "Primary Function": "128-D/512-D Biometric Embedding & Identity Verification",
            "Latency": "~80 - 250 ms (CPU/GPU)",
            "Lighting / Pose Invariance": "Extremely High (Trained on millions of variations)",
            "Strengths": "State-of-the-art identity verification, compact vector comparisons",
            "Weaknesses": "Focuses on identity invariance rather than emotion classification"
        }
    ]

    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        """
        ### 🚀 **Key Architectural Takeaways**
        - **Pipeline Synergy:** Real-world state-of-the-art face recognition systems combine these techniques:
          1. **Viola-Jones or Haar/SSD/MTCNN** is first used to *locate* and crop the face boundary in milliseconds.
          2. **Template Matching** is used for precise pupil/iris tracking or eye alignment.
          3. **FaceNet** computes the normalized compact embedding for instant database lookups.
          4. **DeepFace** predicts behavioral attributes such as emotions, mood, and expressions.
        """
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 15px;">
        FaceVision Studio • Built with Streamlit, OpenCV, Plotly & Deep Learning • Ready for Local & Cloud Deployment
    </div>
    """,
    unsafe_allow_html=True
)
