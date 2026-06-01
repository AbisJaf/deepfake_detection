import streamlit as st
import time
import os
import sys
import tempfile
import torch

# --- FIX IMPORT PATHS ---
# Get the current directory (frontend) and the parent directory (project root)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add the project root to Python's path so it can find the 'models' and 'data' folders
sys.path.append(parent_dir)

# Now these imports will work perfectly!
from models.detector import MultimodalDeepfakeDetector
from data.preprocessing import VisualPreprocessor, AudioPreprocessor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VeriFace // Deepfake Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CACHE THE MODEL (Prevents reloading weights on every run) ---
@st.cache_resource
def load_system(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate Model
    model = MultimodalDeepfakeDetector(embed_dim=512).to(device)
    
    # Load Weights
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        success = True
    else:
        success = False
        
    return model, device, success

# --- CUSTOM BEAUTIFUL THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1a1c29, #0e1017);
        font-family: 'Inter', sans-serif; color: #E2E8F0;
    }
    .main-title {
        font-size: 3rem; font-weight: 700; text-align: center; margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { text-align: center; color: #94A3B8; font-size: 1.1rem; margin-bottom: 3rem; }
    div[data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5); border: 2px dashed #4F46E5 !important;
        border-radius: 16px; padding: 2rem; backdrop-filter: blur(10px);
    }
    .result-card {
        padding: 2rem; border-radius: 16px; text-align: center; margin-top: 2rem;
        backdrop-filter: blur(10px); animation: fadeInUp 0.6s ease-out;
    }
    .fake-result {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(127, 29, 29, 0.4) 100%);
        border: 1px solid #EF4444; box-shadow: 0 0 30px rgba(239, 68, 68, 0.15);
    }
    .real-result {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 78, 59, 0.4) 100%);
        border: 1px solid #10B981; box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
    }
    .status-text { font-size: 2.2rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 0.5rem; }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    default_ckpt = r"D:\fyp\app\deepfake_detection\deepfake_detector_adapted_BEST.pth"
    checkpoint_path = st.text_input("Checkpoint Path (.pth)", value=default_ckpt)
    
    st.markdown("---")
    st.header("🎛️ Calibration")
    # This lets you shift the sensitivity of the model
    decision_threshold = st.slider(
        "Detection Threshold", 
        min_value=0.0, max_value=1.0, value=0.50, step=0.01,
        help="Increase this if the model is too sensitive (calling real videos fake). Check your evaluate.py output for the 'Best threshold' metric and match it here."
    )
    
    st.markdown("---")
    st.caption("VeriFace uses MTCNN for face cropping, 80-band Mel-spectrograms for audio, and Cross-Attention Fusion for deepfake classification.")

# --- UI HEADER ---
st.markdown('<h1 class="main-title">🛡️ VeriFace</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Multimodal Audio-Visual Deepfake Detection System</p>', unsafe_allow_html=True)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader(
    "Drag and drop a video file for analysis", 
    type=["mp4", "avi", "mov", "mkv"]
)

# --- MAIN LOGIC ---
if uploaded_file is not None:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("### 📹 Video Preview")
        st.video(uploaded_file)
        
    with col2:
        st.markdown("### 🧠 Model Analysis")
        status_box = st.empty()
        
        # Load the model
        model, device, is_loaded = load_system(checkpoint_path)
        if not is_loaded:
            st.error(f"❌ Weights not found at `{checkpoint_path}`. Please update the path in the sidebar.")
            st.stop()

        # Save uploaded file temporarily for OpenCV and Librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name

        try:
            # --- 1. INITIALIZE PREPROCESSORS ---
            status_box.info("Initializing preprocessors...")
            vis_prep = VisualPreprocessor(device=str(device))
            aud_prep = AudioPreprocessor()

            # --- 2. VISUAL PROCESSING ---
            status_box.info("Extracting uniform frames & detecting faces (MTCNN)...")
            raw_frames = vis_prep.extract_uniform_frames(temp_video_path)
            if not raw_frames:
                st.error("❌ Could not extract enough frames from the video.")
                st.stop()

            face_crops, _ = vis_prep.process_frames(raw_frames)
            if not face_crops:
                st.error("❌ MTCNN failed to detect faces in the extracted frames.")
                st.stop()
            
            # Create Visual Tensor
            vis_tensors = torch.stack([vis_prep.full_face_transform(f) for f in face_crops])

            # --- 3. AUDIO PROCESSING ---
            status_box.info("Extracting audio tracks & computing Mel-spectrograms...")
            try:
                waveform = aud_prep.process_audio(temp_video_path)
                aud_tensor = aud_prep.waveform_to_mel(waveform)
            except Exception as e:
                st.error(f"❌ Audio processing failed: {e}")
                st.stop()

            # --- 4. INFERENCE ---
            status_box.info("Running Cross-Attention Fusion inference...")
            
            # Add batch dimension and move to device
            visuals = vis_tensors.unsqueeze(0).to(device)
            audios = aud_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                # Based on evaluate.py, the model returns logits (un-squashed)
                logits, _, _ = model(visuals, audios)
                
                # Apply Sigmoid to get probability
                fake_probability = torch.sigmoid(logits).item()

            # Use the slider's threshold instead of a hard 0.5
            is_fake = fake_probability >= decision_threshold

            # --- 5. RENDER RESULTS ---
            status_box.empty() # Clear the loading message
            
            # Display the raw score for debugging purposes
            st.markdown(f"**Diagnostic Data:** Model Raw Fake Probability: `{fake_probability:.4f}` (Threshold: `{decision_threshold}`)")
            
            if is_fake:
                st.markdown(f"""
                    <div class="result-card fake-result">
                        <div class="status-text" style="color: #EF4444;">🚨 ALTERED / FAKE</div>
                        <p style="color: #FCA5A5; font-size: 1.1rem; margin: 0;">
                            Probability of manipulation: <b>{(fake_probability * 100):.2f}%</b>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card real-result">
                        <div class="status-text" style="color: #10B981;">✅ VERIFIED REAL</div>
                        <p style="color: #A7F3D0; font-size: 1.1rem; margin: 0;">
                            Probability of manipulation: <b>{(fake_probability * 100):.2f}%</b> 
                        </p>
                    </div>
                """, unsafe_allow_html=True)

        finally:
            # Clean up the temporary file so it doesn't clutter your drive
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except PermissionError:
                    pass
else:
    st.markdown("---")
    st.caption("<center>Upload a video file to run the deepfake diagnostic pipeline.</center>", unsafe_allow_html=True)



# import streamlit as st
# import time
# import os
# import sys
# import tempfile
# import torch

# # --- FIX IMPORT PATHS ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
# sys.path.append(parent_dir)

# from models.detector import MultimodalDeepfakeDetector
# from data.preprocessing import VisualPreprocessor, AudioPreprocessor

# # --- PAGE CONFIGURATION ---
# st.set_page_config(
#     page_title="VeriFace // Deepfake Detector",
#     page_icon="🛡️",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )

# # --- CACHE THE MODEL ---
# @st.cache_resource
# def load_system(checkpoint_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MultimodalDeepfakeDetector(embed_dim=512).to(device)
#     if os.path.exists(checkpoint_path):
#         model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#         model.eval()
#         return model, device, True
#     return model, device, False

# # --- CUSTOM THEME ---
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
#     html, body, [data-testid="stAppViewContainer"] {
#         background: radial-gradient(circle at top right, #1a1c29, #0e1017);
#         font-family: 'Inter', sans-serif; color: #E2E8F0;
#     }
#     .main-title {
#         font-size: 3rem; font-weight: 700; text-align: center; margin-bottom: 0.5rem;
#         background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
#         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
#     }
#     .subtitle { text-align: center; color: #94A3B8; font-size: 1.1rem; margin-bottom: 3rem; }
#     div[data-testid="stFileUploader"] {
#         background: rgba(30, 41, 59, 0.5); border: 2px dashed #4F46E5 !important;
#         border-radius: 16px; padding: 2rem; backdrop-filter: blur(10px);
#     }
#     .result-card {
#         padding: 2rem; border-radius: 16px; text-align: center; margin-top: 2rem;
#         backdrop-filter: blur(10px); animation: fadeInUp 0.6s ease-out;
#     }
#     .fake-result {
#         background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(127, 29, 29, 0.4) 100%);
#         border: 1px solid #EF4444; box-shadow: 0 0 30px rgba(239, 68, 68, 0.15);
#     }
#     .real-result {
#         background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 78, 59, 0.4) 100%);
#         border: 1px solid #10B981; box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
#     }
#     .status-text { font-size: 2.2rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 0.5rem; }
#     @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
#     </style>
# """, unsafe_allow_html=True)

# # --- SIDEBAR CONFIGURATION ---
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     default_ckpt = r"D:\fyp\app\deepfake_detection\deepfake_detector_adapted_BEST.pth"
#     checkpoint_path = st.text_input("Checkpoint Path (.pth)", value=default_ckpt)

#     st.markdown("---")
#     st.header("🎛️ Calibration")

#     # ⚠️ FIX 1: Default threshold lowered to 0.35.
#     # If your evaluate.py reported a "Best threshold", use that value here instead.
#     # Common fix: models trained on FaceForensics-style data often have best threshold ~0.2–0.4
#     decision_threshold = st.slider(
#         "Detection Threshold",
#         min_value=0.0, max_value=1.0, value=0.35, step=0.01,
#         help="Lower this if everything is being called FAKE. Run evaluate.py and use the 'Best threshold' it reports."
#     )

#     st.markdown("---")
#     st.header("🔬 Debug Mode")
#     # ⚠️ FIX 2: Label convention toggle.
#     # If your training used label=0 for FAKE and label=1 for REAL, tick this box.
#     # If you used label=1 for FAKE and label=0 for REAL, leave it unticked.
#     invert_label = st.checkbox(
#         "Invert label (REAL=1, FAKE=0)",
#         value=False,
#         help="Tick this if your training dataset assigned label=0 to fake videos. This flips the sigmoid output."
#     )
#     show_debug = st.checkbox("Show diagnostic data", value=True)

#     st.markdown("---")
#     st.caption("VeriFace uses MTCNN for face cropping, 80-band Mel-spectrograms for audio, and Cross-Attention Fusion for deepfake classification.")

# # --- UI HEADER ---
# st.markdown('<h1 class="main-title">🛡️ VeriFace</h1>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle">Advanced Multimodal Audio-Visual Deepfake Detection System</p>', unsafe_allow_html=True)

# # --- FILE UPLOADER ---
# uploaded_file = st.file_uploader(
#     "Drag and drop a video file for analysis",
#     type=["mp4", "avi", "mov", "mkv"]
# )

# # --- MAIN LOGIC ---
# if uploaded_file is not None:
#     col1, col2 = st.columns([1.2, 1])

#     with col1:
#         st.markdown("### 📹 Video Preview")
#         st.video(uploaded_file)

#     with col2:
#         st.markdown("### 🧠 Model Analysis")
#         status_box = st.empty()

#         # Load the model
#         model, device, is_loaded = load_system(checkpoint_path)
#         if not is_loaded:
#             st.error(f"❌ Weights not found at `{checkpoint_path}`. Please update the path in the sidebar.")
#             st.stop()

#         with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             temp_video_path = tmp_file.name

#         try:
#             # --- 1. INITIALIZE PREPROCESSORS ---
#             status_box.info("Initializing preprocessors...")
#             vis_prep = VisualPreprocessor(device=str(device))
#             aud_prep = AudioPreprocessor()

#             # --- 2. VISUAL PROCESSING ---
#             status_box.info("Extracting uniform frames & detecting faces (MTCNN)...")
#             raw_frames = vis_prep.extract_uniform_frames(temp_video_path)
#             if not raw_frames:
#                 st.error("❌ Could not extract enough frames from the video.")
#                 st.stop()

#             face_crops, _ = vis_prep.process_frames(raw_frames)
#             if not face_crops:
#                 st.error("❌ MTCNN failed to detect faces in the extracted frames.")
#                 st.stop()

#             vis_tensors = torch.stack([vis_prep.full_face_transform(f) for f in face_crops])

#             # --- 3. AUDIO PROCESSING (with fallback for silent/no-audio videos) ---
#             status_box.info("Extracting audio tracks & computing Mel-spectrograms...")
#             audio_fallback_used = False
#             try:
#                 waveform = aud_prep.process_audio(temp_video_path)
#                 aud_tensor = aud_prep.waveform_to_mel(waveform)
#                 st.write("Audio tensor shape:", aud_tensor.shape)  # ← add this
#                 st.stop()  # ← stop here so you can see the shape
#             except Exception as e:
#                 # ⚠️ FIX 3: Instead of crashing, use a zero tensor as a silent audio placeholder.
#                 # Shape (1, 80, 300) matches a typical 80-band mel-spectrogram for ~3s audio.
#                 # Adjust the last dimension if your model uses a different time length.
#                 st.warning(f"⚠️ Audio extraction failed: `{e}`. Using silent audio placeholder.")
#                 audio_fallback_used = True
#                 # Try to infer the expected audio shape from a dummy forward pass shape
#                 # Default: (batch=1, channels=1, mel_bands=80, time_frames=300)
#                 aud_tensor = torch.zeros(1, 80, 300)

#             # --- 4. INFERENCE ---
#             status_box.info("Running Cross-Attention Fusion inference...")

#             visuals = vis_tensors.unsqueeze(0).to(device)
#             # Force correct shape regardless of what preprocessor returns
#             aud_tensor = aud_tensor.reshape(1, 80, 300)  # remove all extra dims
#             audios = aud_tensor.unsqueeze(0).to(device)  # → [1, 1, 80, 300]

#             with torch.no_grad():
#                 logits, _, _ = model(visuals, audios)
#                 raw_sigmoid = torch.sigmoid(logits).item()

#             # ⚠️ FIX 2 (applied): If label convention is REAL=1/FAKE=0, sigmoid gives P(real).
#             # The checkbox in sidebar lets you flip this without touching code.
#             if invert_label:
#                 fake_probability = 1.0 - raw_sigmoid
#             else:
#                 fake_probability = raw_sigmoid

#             is_fake = fake_probability >= decision_threshold

#             # --- 5. RENDER RESULTS ---
#             status_box.empty()

#             # ⚠️ Debug panel — helps during demo to explain the numbers to supervisors
#             if show_debug:
#                 st.markdown("**Diagnostic Data:**")
#                 st.write({
#                     "Raw logit": f"{logits.item():.4f}",
#                     "Raw sigmoid": f"{raw_sigmoid:.4f}",
#                     "Fake probability (after label flip)": f"{fake_probability:.4f}",
#                     "Threshold used": decision_threshold,
#                     "Label inverted": invert_label,
#                     "Audio fallback used": audio_fallback_used,
#                     "Visual frames": len(face_crops),
#                     "Device": str(device),
#                 })

#             if is_fake:
#                 st.markdown(f"""
#                     <div class="result-card fake-result">
#                         <div class="status-text" style="color: #EF4444;">🚨 ALTERED / FAKE</div>
#                         <p style="color: #FCA5A5; font-size: 1.1rem; margin: 0;">
#                             Probability of manipulation: <b>{(fake_probability * 100):.2f}%</b>
#                         </p>
#                     </div>
#                 """, unsafe_allow_html=True)
#             else:
#                 st.markdown(f"""
#                     <div class="result-card real-result">
#                         <div class="status-text" style="color: #10B981;">✅ VERIFIED REAL</div>
#                         <p style="color: #A7F3D0; font-size: 1.1rem; margin: 0;">
#                             Probability of manipulation: <b>{(fake_probability * 100):.2f}%</b>
#                         </p>
#                     </div>
#                 """, unsafe_allow_html=True)

#         finally:
#             if os.path.exists(temp_video_path):
#                 try:
#                     os.remove(temp_video_path)
#                 except PermissionError:
#                     pass
# else:
#     st.markdown("---")
#     st.caption("<center>Upload a video file to run the deepfake diagnostic pipeline.</center>", unsafe_allow_html=True)