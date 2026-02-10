from huggingface_hub import hf_hub_download
import os
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Veritas AI | Multimodal Fake News Detector",
    page_icon="🕵️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Added Weights Box Styling)
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        font-size: 20px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover { background-color: #ff3333; color: white; }
    .verdict-real { color: #28a745; font-weight: bold; font-size: 40px; text-align: center; }
    .verdict-fake { color: #dc3545; font-weight: bold; font-size: 40px; text-align: center; }

    /* Sidebar Weights Box Styling */
    .weights-box {
        background-color: #e0f2fe; /* Light Blue Bg */
        border: 1px solid #7dd3fc;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .weights-title {
        color: #0369a1;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 10px;
        display: block;
    }
    .weight-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0c4a6e;
        display: block;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (Hugging Face Version)
# ==========================================
@st.cache_resource
def load_models():
    MODEL_REPO = "RUTWIK55/veritas-fake-news-models"
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- A. Download & Load Text Model (.h5) ---
    try:
        text_model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="final_text_model.h5",
            local_dir=MODEL_DIR
        )

        text_model = tf.keras.models.load_model(
            text_model_path,
            custom_objects={"TFBertModel": TFBertModel}
        )
        print("✅ Text Model Loaded from Hugging Face")
    except Exception as e:
        st.error(f"❌ Text Model Error: {e}")
        text_model = None

    # --- B. Download & Load Image Model (.keras) ---
    try:
        image_model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="final_image_model.keras",
            local_dir=MODEL_DIR
        )

        image_model = keras.models.load_model(image_model_path)
        print("✅ Image Model Loaded from Hugging Face")
    except Exception as e:
        st.error(f"❌ Image Model Error: {e}")
        image_model = None

    return text_model, image_model


# Load Resources
text_model, image_model = load_models()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964063.png", width=100)
    st.title("Settings")
    st.markdown("### Fusion Balance")
    
    # Single slider controls both weights
    # Value 0.0 = 100% Image, 1.0 = 100% Text
    balance = st.slider("Text vs. Image Importance", 0.0, 1.0, 0.6, 0.05)
    
    # Calculate weights automatically so they ALWAYS sum to 1.0
    w_text = balance
    w_image = 1.0 - balance
    
    # --- CUSTOM CENTERED WEIGHTS BLOCK ---
    st.markdown(f"""
    <div class="weights-box">
        <span class="weights-title">Current Weights</span>
        <span class="weight-value">📝 Text: {w_text:.2f}</span>
        <span class="weight-value">🖼️ Image: {w_image:.2f}</span>
    </div>
    """, unsafe_allow_html=True)
    # -------------------------------------

    st.caption("Architecture: BERT + ResNet50")

# ==========================================
# 4. MAIN UI
# ==========================================
st.title("🕵️‍♀️ Veritas AI | Multimodal Fake News Detector")
st.markdown("##### Detect misinformation using **BERT** (Text) and **ResNet50** (Image).")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Enter News Details")
    headline = st.text_area("News Headline", placeholder="e.g., Aliens land in New York City...", height=100)
    uploaded_file = st.file_uploader("Upload News Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
    verify_btn = st.button("🔍 Verify Authenticity")

with col2:
    st.subheader("2. Analysis Report")
    
    if verify_btn:
        if not headline or not uploaded_file:
            st.error("⚠️ Please provide both a headline and an image.")
        elif text_model is None or image_model is None:
            st.error("⚠️ Models failed to load. Check console.")
        else:
            with st.spinner("Processing with BERT & ResNet50..."):
                # --- PREDICTION LOGIC ---
                
                # A. Text (BERT)
                encodings = tokenizer([headline], truncation=True, padding='max_length', 
                                      max_length=64, return_tensors='tf')
                
                # IMPORTANT: Functional models require specific input names
                p_text_fake = text_model.predict({
                    'input_ids': encodings['input_ids'], 
                    'attention_mask': encodings['attention_mask']
                }, verbose=0)[0][0]
                
                # B. Image (ResNet50)
                img = image.resize((160, 160)) 
                img_array = np.array(img)
                if img_array.shape[-1] == 4: img_array = img_array[..., :3]
                
                img_array = preprocess_input(img_array)
                img_array = tf.expand_dims(img_array, 0)
                
                img_preds = image_model.predict(img_array, verbose=0)[0]
                p_image_fake = img_preds[1] 
                
                # C. Fusion
                p_final = (w_text * p_text_fake) + (w_image * p_image_fake)
                
                label = "FAKE NEWS" if p_final > 0.5 else "REAL NEWS"
                confidence = p_final if p_final > 0.5 else 1 - p_final
                
                if p_final > 0.5:
                    # Case: FAKE (High Probability) -> Show Red Alert
                    st.markdown(f'<div class="verdict-fake">🚨 {label}</div>', unsafe_allow_html=True)
                else:
                    # Case: REAL (Low Probability) -> Show Green Check
                    st.markdown(f'<div class="verdict-real">✅ {label}</div>', unsafe_allow_html=True)
                
                st.progress(float(confidence), text=f"Confidence: {confidence:.1%}")
                st.markdown("---")
                
                # Breakdown
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("BERT Score (Fake Prob)", f"{p_text_fake:.2f}")
                with c2:
                    st.metric("ResNet Score (Fake Prob)", f"{p_image_fake:.2f}")

    else:

        st.info("Waiting for input...")



