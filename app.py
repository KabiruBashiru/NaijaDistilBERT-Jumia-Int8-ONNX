import streamlit as st
import torch
import numpy as np
import time
import os
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from huggingface_hub import hf_hub_download, list_repo_files  # Added list_repo_files

# --- Page Configuration ---
st.set_page_config(
    # page_title="Sentiment Analysis",
    page_title="Jumia-Senti",
    page_icon="💬",
    layout="centered"
)

# st.title("💬 Sentiment Analysis on Naija Pidgin")
st.title("💬 Jumia-Senti: Sentiment Analysis on Product Review")
st.markdown("Enter product review below to analyze sentiment using our state-of-the-art transformer model")

# --- Model Configuration ---
MODEL_ID = "BinBashir/NaijaDistilBERT-Jumia-Int8-ONNX"

@st.cache_resource
def get_model():
    """
    Loads the Tokenizer and the ONNX Model.
    """
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    

    # On Streamlit Cloud, always stick to CPU to avoid shared library errors
    # unless you are on a private paid tier with GPU.
    model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        provider="CPUExecutionProvider"
    
    )
    return tokenizer, model

# --- Load Model ---
try:
    with st.spinner(f"Loading {MODEL_ID} Model..."):
        tokenizer, model = get_model()

    # --- Model Size Logic (Fixed) ---
    try:
        # 1. Get list of all files in the Hugging Face repo
        repo_files = list_repo_files(repo_id=MODEL_ID)
        
        # 2. Find the first file that ends with '.onnx' 
        # (This finds 'model.onnx', 'model_quantized.onnx', etc. automatically)
        onnx_filename = next((f for f in repo_files if f.endswith('.onnx')), None)
        
        if onnx_filename:
            # 3. Download that specific file path to calculate size
            model_weights_path = hf_hub_download(
                repo_id=MODEL_ID,
                filename=onnx_filename
            )
            
            file_size = os.path.getsize(model_weights_path) 
            file_size_mb = file_size / (1024 * 1024)
            
            st.success("Model Loaded Successfully!")
            st.metric(label=f"💾 Model Size ({onnx_filename})", value=f"{file_size_mb:.2f} MB")
        else:
            st.warning("Model loaded, but could not find an .onnx file to calculate size.")
            
    except Exception as e:
        # If size calculation fails, just show a warning but keep the app running
        st.warning(f"Could not calculate size (Internet/API error): {e}")
        st.success("Model Loaded Successfully!")

except Exception as e:
    st.error(f"Error loading model. Please check the Model ID or libraries.\nDetails: {e}")
    st.stop()

# --- User Input ---
st.divider()
user_input = st.text_area('Enter Text to Analyze')

button = st.button("Analyze Sentiment", type="primary", use_container_width=True)

# Label Mapping
d = {
    0: 'NEUTRAL 😐',
    1: 'POSITIVE 😊',
    2: 'NEGATIVE 😞'
}

# --- Inference Logic ---
if user_input and button:
    # 1. Tokenize Input
    inputs = tokenizer(
        user_input, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors='np'
    )

    # 2. Inference & Timing
    start_time = time.time()

    #with torch.no_grad():
    output = model(**inputs)

    end_time = time.time()
    inference_time = end_time - start_time

    # 3. Process Results
    logits = output.logits
    
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
        
    y_pred = int(np.argmax(logits, axis=1)[0])
    
    # Display Result
    result_label = d.get(y_pred, f"Unknown Label ({y_pred})")
    
    st.success(f"Prediction: {result_label}")

    st.metric(label="⏱ Inference Time", value=f"{inference_time:.4f} seconds")

    st.info(f"⚡ Inference Speed ⏱ : {inference_time:.4f} seconds")
